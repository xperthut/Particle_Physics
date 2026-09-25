"""Generate M(omega, P) matrix elements from a fitted I(omega) ITD curve.

Two stages, both per lattice ensemble (``FM`` = 09 / 12 / 15):

1. ``create_combined_data`` -- concatenate the raw per-momentum RpITD files
   into ``Data/2026/All_p_w_m.csv`` (with a single omega=0 anchor per (FM, P)).

2. ``generate_m_omega_from_I_omega`` -- for each FM:
     a. read the fitted I(omega) line and interpolate it onto the real omega
        grid of that ensemble;
     b. fit mi = M(omega, P=i) and mj = M(omega, P=j) where j>i>0, by gradient descent on
            I = ((j^2 * mj) - (i^2 * mi)) / (j^2 - i^2)
        seeded from ``mi = alpha * I(omega)``, ``mj = beta``;
     c. derive the higher moments (mj, j>i) from mi via
            mj = ((j^2 - i^2) * I + (i^2*mi)) / j^2 ;
     d. search for the (alpha, beta) that minimise the absolute difference
        between the generated and the real mean M at the matched omega values.

Run with ``Helicity/Code`` as the working directory:  ``python synthetic_data_generator.py``
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

mpl.rcParams['figure.figsize'] = (12, 10)
mpl.rcParams.update({'font.size': 22})

SMOOTH_WINDOW = 60       # Savitzky-Golay window in grid points (0.1 each)
SMOOTH_POLYORDER = 3

VERSION = 2026
DATA_DIR = f'../Data/{VERSION}/'
IMG_DIR = f'../Data/{VERSION}/Images/'

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

def get_smooth_curve(val, window=SMOOTH_WINDOW, polyorder=SMOOTH_POLYORDER):
    """Savitzky-Golay smooth of a 1-D series on a uniform grid.

    The window is forced odd and shrunk to fit short inputs; if it cannot be
    made larger than ``polyorder`` the series is returned unchanged.
    """
    val = np.asarray(val, dtype=float)
    n = val.size
    w = min(int(window), n if n % 2 == 1 else n - 1)
    if w <= polyorder or w < 5:
        return val
    return savgol_filter(val, window_length=w, polyorder=polyorder, mode='interp')

# I(omega) = B*mj + A*mi for the directly-fit momentum pair (i, j), j>i>=1,
# with A = -i^2/(j^2-i^2), B = j^2/(j^2-i^2). (i, j) = (1, 2) -- A=-1/3,
# B=4/3 -- is this repo's default pair everywhere below.
def _ab_for_pair(i, j):
    """(A, B) coefficients of I = A*mi + B*mj for momentum pair i<j."""
    denom = float(j ** 2 - i ** 2)
    return -(i ** 2) / denom, (j ** 2) / denom

# The real omega grid is irregular (multiples of pz * const) while the fitted
# ITD line is sampled every 0.1.  Round both to one decimal and compare as
# integers so the merge in stage 2 is exact.
W_SCALE = 10

# Gradient-descent hyper-parameters for the mi/mj fit.
LR = 0.01
EPOCHS = 100


def _w_key(w):
    """Integer key for merging omega values (see W_SCALE)."""
    return np.round(np.asarray(w, dtype=float) * W_SCALE).astype(int)


def create_combined_data():
    """Build ``Data/2026/All_p_w_m.csv`` from the raw RpITD text files."""
    data = None
    for fm in ['09', '12', '15']:
        for p in range(1, 6):
            df = pd.read_csv(
                os.path.join(DATA_DIR, 'rpitd', f'rpitd_a{fm}m310_Wilson3_pz{p}.txt'),
                sep=' ', skiprows=2, header=None,
            )
            df.columns = ['Exp', 'Z', 'W', 'M']
            df['P'] = p
            df['FM'] = int(fm)

            data = df if data is None else pd.concat([data, df], ignore_index=True)

    # Add a single W=0, M=0 anchor row per (FM, P, Exp) -- omega=0 corresponds to
    # Z=0 (no Wilson-line displacement), so it must be added once per (FM, P),
    # NOT once per observed Z value (that previously duplicated the anchor block
    # zmax times per (FM,P) -- e.g. 10x for FM=09/P=1 -- doubling All_p_w_m.csv's
    # row count and pairing W=0 with nonsensical nonzero Z labels).
    exp_counts = data.groupby(['FM', 'P'])['Exp'].nunique().reset_index(name='Exp_count')

    for _, row in exp_counts.iterrows():
        fm, p, n = row['FM'], row['P'], row['Exp_count']
        anchor = pd.DataFrame({
            'FM': fm, 'P': p, 'Z': 0, 'Exp': range(0, n), 'W': 0.0, 'M': 0.0,
        })
        data = pd.concat([data, anchor], ignore_index=True)

    data.sort_values(['FM', 'P', 'Z', 'W', 'Exp', 'M'], ascending=True, inplace=True)

    print('Data loaded successfully!')
    print(data.shape)
    print(data.head())

    data.to_csv(os.path.join(DATA_DIR, 'All_p_w_m.csv'), index=False)
    return data


def real_mean_M(data, fm):
    """Mean/std real M per (P, W) for one ensemble, with an integer omega key."""
    df_g = (
        data[data.FM == fm]
        .groupby(['P', 'W'])['M'].agg(Mean_M='mean', Std_M='std')
        .reset_index()
    )
    df_g['Std_M'] = df_g['Std_M'].fillna(0.0)
    df_g['Wkey'] = _w_key(df_g['W'])
    return df_g


def plot_p_w_m_curves(data=None):
    if data is None:
        data = pd.read_csv(os.path.join(DATA_DIR, 'All_p_w_m.csv'))

    for fm in data['FM'].unique():
        df_g = real_mean_M(data, fm)

        plt.figure(dpi=300)
        for p in sorted(df_g.P.unique()):
            t = df_g[df_g.P == p]
            plt.plot(t.W, t.Mean_M, '-o', label=fr"$p_z$={p}")

        plt.xlabel(r'$\omega$')
        plt.ylabel(r'$\Delta\mathfrak{M}(\omega)$')
        plt.title(f'FM={fm}')
        plt.legend()
        plt.savefig(os.path.join(IMG_DIR, f'P_W_M_curves_FM_{fm}.pdf'), format='pdf', dpi=300)
        plt.close()


def plot_I_omega(df_I):
    plt.figure(dpi=300)
    plt.plot(df_I.W, df_I.I, '-o')
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\Delta \mathcal{I}(\omega)$')
    plt.savefig(os.path.join(IMG_DIR, 'I(w).pdf'), format='pdf', dpi=300)
    plt.close()


def fit_mi_mj(I_omega, alpha, beta, i=1, j=2, record_loss=False, omega=None):
    """Gradient-descent fit of ``mi``, ``mj`` -- the directly-fit momentum
    pair, ``j>i>=1``, default ``(i, j) = (1, 2)`` -- to
    ``I = A*mi + B*mj`` for each omega, with
    ``A = -i^2/(j^2-i^2)``, ``B = j^2/(j^2-i^2)`` (see ``_ab_for_pair``).

    ``I_omega`` is a 1-D array of I values.  The seed is ``mi = alpha * I``,
    ``mj = beta * I``.  All omega points are optimised together (they are
    independent).  Returns ``(mi, mj)`` arrays; if ``record_loss`` also returns
    a long-form DataFrame with per-epoch loss (needs ``omega`` for labelling).

    ``record_loss`` defaults to False: every caller except the one explicit
    "per-epoch loss trace" call in ``generate_m_omega_from_I_omega`` (which
    passes both ``record_loss=True`` and ``omega=``) relies on this default --
    notably ``estimate_M`` and ``_fit``'s ``objective`` (the latter runs this
    thousands of times per optimisation), neither of which pass ``omega``, so
    a True default crashes them (``zip(None, ...)``).

    Both seeds scale with ``I`` (not just ``mi``'s) so the whole trajectory
    -- not just the endpoint -- passes through (0, 0) as ``I -> 0``: a flat
    ``mj = beta`` seed stays at ``beta`` regardless of how small ``I`` is,
    so with the limited (100-epoch) convergence budget below, nearby-omega
    points land close to that nonzero seed, producing a visible jump against
    ``estimate_M``'s explicit ``M(omega=0)=0`` anchor. Proportional seeding
    removes the discontinuity at its source rather than patching the one
    boundary point.
    """
    Ai, Bj = _ab_for_pair(i, j)
    y = np.asarray(I_omega, dtype=float)
    mi = alpha * y
    mj = beta * y

    loss_rows = [] if record_loss else None
    for ep in range(1, EPOCHS):
        pred = Ai * mi + Bj * mj
        resid = y - pred                       # (Yreal - Ypred)
        if record_loss:
            for w_i, l_i in zip(omega, resid ** 2):
                loss_rows.append({'epoch': ep, 'w': w_i, 'loss': l_i})
        mi = mi - LR * (-2.0 * Ai * resid)
        mj = mj - LR * (-2.0 * Bj * resid)

    if record_loss:
        return mi, mj, pd.DataFrame(loss_rows)
    return mi, mj


def fit_m1_m2(I_omega, alpha, beta, record_loss=False, omega=None):
    """Backward-compatible alias for ``fit_mi_mj`` with the default (i, j) = (1, 2)."""
    return fit_mi_mj(I_omega, alpha, beta, 1, 2, record_loss, omega)


def estimate_M(omega, I_omega, n_p, alpha, beta, gamma=0.0, i=1, j=2):
    """Long-form generated M: columns ``W``, ``P`` (1..n_p), ``Est_M``, ``Wkey``.

    ``i, j`` (default 1, 2) -- the directly-fit momentum pair: ``mi``, ``mj``
    come from ``fit_mi_mj``; every other P's moment is the closed form
    ``m_k = ((k^2-i^2)*I + i^2*mi) / k^2``, the algebraic inverse of
    ``I = (k^2*m_k - i^2*mi)/(k^2-i^2)`` for *every* k -- so with gamma=0 any
    two P's reconstruct the same I(omega), a physical requirement (I(omega)
    must not depend on which momentum pair estimated it).

    ``gamma`` -- damping of the derived (P not in {i, j}) moments (option A).
    The closed form tends to ``I(omega)`` as k grows past ``j``, which
    over-predicts the real data at high P.  ``gamma`` multiplies each derived
    moment by ``exp(-gamma * (k - j))`` so the tail can be pulled down
    without touching the directly-fit ``mi``, ``mj``.  ``gamma=0`` is the
    original ansatz.
    """
    mi, mj = fit_mi_mj(I_omega, alpha, beta, i, j)
    y = np.asarray(I_omega, dtype=float)

    # M(omega=0, P) = 0 exactly by construction (I(0) = 0). fit_mi_mj's seeds
    # are both proportional to I, so this is already ~0 up to the GD's
    # 100-epoch convergence residual; this mask just makes it exact.
    zero = y == 0.0
    mi = np.where(zero, 0.0, mi)
    mj = np.where(zero, 0.0, mj)

    cols = {'W': np.asarray(omega, dtype=float), i: mi, j: mj}
    for k in range(1, n_p + 1):
        if k in (i, j):
            continue
        cols[k] = (((k ** 2 - i ** 2) * y + (i ** 2) * mi) / (k ** 2)) * np.exp(-gamma * (k - j))

    df = pd.DataFrame(cols).melt(id_vars='W', var_name='P', value_name='Est_M')
    df['P'] = df['P'].astype(int)
    df['Wkey'] = _w_key(df['W'])
    return df


def _p_weights(df_real_g, p_weight_exp):
    """Per-point weight  w = (1 / RMS_P) * P**p_weight_exp  (option D).

    ``1/RMS_P`` puts every momentum on an equal footing regardless of its
    magnitude; ``P**p_weight_exp`` then tilts the fit toward higher P
    (exp=0 -> equal per-P, exp=1/2 -> progressively favour large P).
    """
    rms = df_real_g.groupby('P')['Mean_M'].apply(lambda s: float(np.sqrt(np.mean(s ** 2))))
    return df_real_g['P'].map(lambda p: (1.0 / rms[p]) * float(p) ** p_weight_exp).to_numpy()


def match_real_and_estimated_M(df_real_g, df_est, weights=None):
    """(weighted) sum of |Mean_M - Est_M| over shared omega values, plus the count.

    ``weights`` -- optional array aligned with ``df_real_g`` rows; when given the
    absolute differences are weighted by it before summing.
    """
    real = df_real_g[['P', 'Wkey', 'Mean_M']].copy()
    real['_w'] = 1.0 if weights is None else np.asarray(weights, dtype=float)
    merged = pd.merge(real, df_est, on=['P', 'Wkey'], how='inner')
    diff = float(np.sum(merged['_w'] * np.abs(merged['Mean_M'] - merged['Est_M'])))
    return diff, len(merged)


def _fit(df_real_g, omega, I_omega, n_p, seeds, use_gamma, weights, monotonic_penalty=0.0,
         i=1, j=2):
    """Multi-start Nelder-Mead core shared by the fit_* functions.

    ``i, j`` (default 1, 2) -- the directly-fit momentum pair (see ``estimate_M``).

    The objective is evaluated with plain numpy (no per-iteration DataFrame
    build) since it runs thousands of times.

    ``monotonic_penalty`` (0 = off) adds ``penalty * sum(max(0, rel[p+1] -
    rel[p])**2)`` over consecutive P, where ``rel[p]`` is P's RMS-relative
    error (as in ``_per_p_rel_rms``) -- i.e. it pushes the search away from
    any (alpha, beta) where a *higher* P fits worse than the P below it.
    Needed because P not in {i, j} are algebraically derived from mi/I with
    no separate free parameter (see ``generate_m_omega_from_I_omega``'s
    docstring): the per-P weight alone shifts the P=i-vs-P=j balance but
    cannot on its own guarantee a strict best-at-highest-P ordering overall.
    """
    y = np.asarray(I_omega, dtype=float)
    y_zero = y == 0.0
    key_to_pos = {int(k): idx for idx, k in enumerate(_w_key(omega))}

    # df_real_g carries a RangeIndex (real_mean_M resets it), so weights[] aligns
    # by position and survives the Wkey filter below.
    mask = df_real_g['Wkey'].isin(key_to_pos).to_numpy()
    real = df_real_g[mask]
    pos = real['Wkey'].map(key_to_pos).to_numpy()          # omega-grid index per real point
    P = real['P'].to_numpy(dtype=float)
    real_M = real['Mean_M'].to_numpy(dtype=float)
    w = np.ones(mask.sum()) if weights is None else np.asarray(weights, dtype=float)[mask]
    P_sorted = np.sort(np.unique(P))
    p_masks = [P == p for p in P_sorted] if monotonic_penalty > 0.0 else None

    def objective(params):
        if use_gamma:
            alpha, beta, gamma = params
        else:
            alpha, beta = params
            gamma = 0.0
        if len(pos) == 0:
            return np.inf
        mi, mj = fit_mi_mj(y, alpha, beta, i, j)
        mi = np.where(y_zero, 0.0, mi)[pos]
        mj = np.where(y_zero, 0.0, mj)[pos]
        yi = y[pos]
        derived = ((P ** 2 - i * i) * yi + (i * i) * mi) / (P ** 2) * np.exp(-gamma * (P - j))
        est = np.where(P == float(i), mi, np.where(P == float(j), mj, derived))
        total = float(np.sum(w * np.abs(real_M - est)))

        if monotonic_penalty > 0.0:
            rel = np.empty(len(P_sorted))
            for idx, m in enumerate(p_masks):
                denom = np.sqrt(np.mean(real_M[m] ** 2))
                rel[idx] = np.sqrt(np.mean((real_M[m] - est[m]) ** 2)) / denom if denom > 0 else 0.0
            violation = np.maximum(0.0, rel[1:] - rel[:-1])
            total += monotonic_penalty * float(np.sum(violation ** 2))
        return total

    best = None
    for x0 in seeds:
        res = minimize(objective, x0=list(x0), method='Nelder-Mead',
                       options={'xatol': 1e-4, 'fatol': 1e-12, 'maxiter': 4000})
        if best is None or res.fun < best.fun:
            best = res
    return best


def fit_alpha_beta(df_real_g, omega, I_omega, n_p, i=1, j=2):
    """Baseline: (alpha, beta) minimising the unweighted sum |Mean_M - Est_M|.

    ``i, j`` (default 1, 2) -- the directly-fit momentum pair (see ``estimate_M``).
    """
    # beta now scales I (fit_mi_mj's mj seed is beta*I, not a flat constant),
    # so it plays the same role as alpha and needs the same seed range.
    seeds = [(a, b) for a in (1.0, 10.0, 50.0, 310.0, 1000.0) for b in (1.0, 10.0, 50.0, 310.0, 1000.0)]
    best = _fit(df_real_g, omega, I_omega, n_p, seeds, use_gamma=False, weights=None, i=i, j=j)
    return float(best.x[0]), float(best.x[1]), float(best.fun)


def fit_alpha_beta_weighted(df_real_g, omega, I_omega, n_p, p_weight_exp=0.5, i=1, j=2):
    """Option D: (alpha, beta) under the per-P weight ``(1/RMS_P) * P**p_weight_exp``.

    Same 2-parameter ansatz as ``fit_alpha_beta`` -- only the objective weighting
    changes, so this shows how far re-weighting alone can shift the fit toward
    high P. ``i, j`` (default 1, 2) -- the directly-fit momentum pair.
    """
    weights = _p_weights(df_real_g, p_weight_exp)
    seeds = [(a, b) for a in (1.0, 10.0, 50.0, 310.0, 1000.0) for b in (1.0, 10.0, 50.0, 310.0, 1000.0)]
    best = _fit(df_real_g, omega, I_omega, n_p, seeds, use_gamma=False, weights=weights, i=i, j=j)
    return float(best.x[0]), float(best.x[1]), float(best.fun)


def fit_alpha_beta_monotonic(df_real_g, omega, I_omega, n_p, p_weight_exp=0.5, monotonic_penalty=200.0,
                              i=1, j=2):
    """(alpha, beta) minimising the P-weighted L1 error subject to a strict
    best-fit-at-highest-P ordering (see ``_fit``'s ``monotonic_penalty``).

    ``i, j`` (default 1, 2) -- the directly-fit momentum pair. P-weighting
    alone (``fit_alpha_beta_weighted``) shifts the P=i-vs-P=j balance but
    does not reliably make the rest of the P's relative error decrease
    monotonically -- empirically it can go either way depending on alpha's
    scale, since those moments are a fixed function of mi/I with no
    separate free parameter. This adds an explicit penalty for any ordering
    violation so the search is pushed toward a genuinely monotonic fit.
    """
    weights = _p_weights(df_real_g, p_weight_exp)
    seeds = [(a, b) for a in (1.0, 5.0, 10.0, 20.0, 50.0, 100.0)
             for b in (-50.0, -20.0, -5.0, 0.0, 5.0, 20.0, 50.0)]
    best = _fit(df_real_g, omega, I_omega, n_p, seeds, use_gamma=False, weights=weights,
                monotonic_penalty=monotonic_penalty, i=i, j=j)
    return float(best.x[0]), float(best.x[1]), float(best.fun)


def fit_alpha_beta_gamma(df_real_g, omega, I_omega, n_p, p_weight_exp=0.0, i=1, j=2):
    """Option A: (alpha, beta, gamma) with the high-P damping term in ``estimate_M``.

    Uses the per-P weight (default exp=0, i.e. 1/RMS_P) so gamma is driven by the
    high-P shape rather than swamped by the large P=i residuals. ``i, j``
    (default 1, 2) -- the directly-fit momentum pair.
    """
    weights = _p_weights(df_real_g, p_weight_exp)
    seeds = [(a, b, g) for a in (10.0, 50.0, 300.0) for b in (10.0, 50.0, 300.0) for g in (0.0, 0.3, 0.8)]
    best = _fit(df_real_g, omega, I_omega, n_p, seeds, use_gamma=True, weights=weights, i=i, j=j)
    return float(best.x[0]), float(best.x[1]), float(best.x[2]), float(best.fun)


def _fit_title(fm, alpha, beta, gamma):
    t = fr'FM={fm}   $\alpha$={alpha:.3g}, $\beta$={beta:.3g}'
    return t + fr', $\gamma$={gamma:.3g}' if gamma else t


def plot_real_vs_synthetic_combined(df_g, est_full, fm, alpha, beta, full_range=False,
                                    gamma=0.0, tag=''):
    """All P on one axes: real mean M (+/- std) points vs synthetic M(omega) curve,
    one colour per P.  ``_full`` variant spans omega 0..20, else zoom to real range.
    ``tag`` is appended to the output filename (e.g. a fit-method label)."""
    all_p = sorted(df_g.P.unique())
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plt.figure(dpi=300)
    w_max = df_g['W'].max()
    for i, p in enumerate(all_p):
        c = colors[i % len(colors)]
        r = df_g[df_g.P == p].sort_values('W')
        s = est_full[est_full.P == p].sort_values('W')
        if not full_range:
            s = s[s.W <= w_max * 1.15]

        plt.plot(s.W, s.Est_M, '-', color=c, lw=2, label=fr"synthetic $p_z$={p}")
        plt.errorbar(r.W, r.Mean_M, yerr=r.Std_M, fmt='^', color=c, ms=7, capsize=3,
                     label=fr"real $p_z$={p}")

    if not full_range:
        plt.xlim(-0.2, w_max * 1.15)

    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\Delta\mathfrak{M}(\omega)$')
    plt.title(_fit_title(fm, alpha, beta, gamma))
    plt.legend(fontsize=10, ncol=2)
    suffix = '_full' if full_range else ''
    plt.savefig(os.path.join(IMG_DIR, f'M_real_vs_synthetic_combined_FM_{fm}{tag}{suffix}.pdf'),
                format='pdf', dpi=300)
    plt.close()


def plot_real_vs_synthetic(df_g, est_full, fm, alpha, beta, full_range=False, gamma=0.0, tag=''):
    """One panel per P: real mean M (+/- std) points vs the synthetic M(omega) curve.

    ``df_g``     -- real_mean_M() output for this ensemble (sparse omega grid).
    ``est_full`` -- estimate_M() output on the full ITD omega grid (0..20).
    ``full_range`` -- if True show omega up to 20, else zoom to the real data range.
    ``tag`` is appended to the output filename (e.g. a fit-method label), same as
    ``plot_real_vs_synthetic_combined``.
    """
    all_p = sorted(df_g.P.unique())
    ncols = 3
    nrows = int(np.ceil(len(all_p) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows),
                             dpi=300, squeeze=False)

    for ax, p in zip(axes.flat, all_p):
        r = df_g[df_g.P == p].sort_values('W')
        s = est_full[est_full.P == p].sort_values('W')

        ax.plot(s.W, s.Est_M, '-', color='tab:orange', lw=2, label='synthetic', zorder=1)
        ax.errorbar(r.W, r.Mean_M, yerr=r.Std_M, fmt='^', color='tab:blue', ms=7,
                    capsize=3, label='real', zorder=2)

        if not full_range:
            ax.set_xlim(-0.2, r.W.max() * 1.15)
            in_view = s[s.W <= r.W.max() * 1.15]
            lo = min(r.Mean_M.min(), in_view.Est_M.min())
            hi = max(r.Mean_M.max(), in_view.Est_M.max())
            pad = 0.1 * (hi - lo or 1.0)
            ax.set_ylim(lo - pad, hi + pad)

        ax.set_title(fr'$p_z$={p}')
        ax.set_xlabel(r'$\omega$')
        ax.set_ylabel(r'$\Delta\mathfrak{M}(\omega)$')
        ax.legend(fontsize=11)

    for ax in axes.flat[len(all_p):]:
        ax.set_visible(False)

    fig.suptitle(_fit_title(fm, alpha, beta, gamma))
    fig.tight_layout()
    suffix = '_full' if full_range else ''
    fig.savefig(os.path.join(IMG_DIR, f'M_real_vs_synthetic_FM_{fm}{tag}{suffix}.pdf'),
                format='pdf', dpi=300)
    plt.close(fig)


def generate_m_omega_from_I_omega(fit_method='baseline', p_weight_exp=0.5, monotonic_penalty=200.0,
                                  i=1, j=2):
    """Fit per FM and write the generated M(omega) tables/plots.

    ``i, j`` (default 1, 2, ``j>i>=1``) -- the directly-fit momentum pair for
    every FM (see ``estimate_M``); every other P's moment is derived from it.

    ``fit_method`` -- ``'baseline'`` (unweighted (alpha, beta), option in
    ``fit_alpha_beta``), ``'weighted'`` (option D -- per-P weighting, uses
    ``p_weight_exp``), ``'monotonic'`` (``'weighted'`` plus an explicit penalty
    enforcing a strict best-at-highest-P fit ordering, ``fit_alpha_beta_monotonic``)
    or ``'gamma'`` (option A -- adds the high-P damping term).

    All keep ``gamma=0`` for the *derived* (P not in {i, j}) moments except
    ``'gamma'`` itself: with gamma=0, every derived moment is exactly
    m_k = ((k^2-i^2)*I + i^2*mi) / k^2, the algebraic inverse of
    I = (k^2*m_k - i^2*mi)/(k^2-i^2) for *every* k -- so any two P's
    reconstruct the *same* I(omega) (same value, same sign as the fitted
    curve), which is a physical requirement (I(omega) must not depend on
    which momentum pair estimated it). ``fit_method='gamma'`` damps the
    derived moments to better match the real per-P M magnitude (see
    ``estimate_M``), but that damping breaks this cross-pair consistency for
    every pair except (i, j) -- use it only for the ``diagnose_fit_methods``
    comparison, not for ``new_M.csv``.

    Default is ``'baseline'``. ``'weighted'`` (and, more aggressively,
    ``'monotonic'``) exist to prioritise higher-P fit quality over P=i, but
    for the (i, j) = (1, 2) default, 2 of this repo's 3 ensembles (FM=12,
    FM=15) push the 2-free-parameter (alpha, beta) search into a
    *degenerate* region where the whole P=1 curve (and sometimes others too)
    flips to M <= 0 everywhere -- e.g. FM=12 weighted: P=1 in [-0.70, 0.0],
    153% relative error -- rather than genuinely fitting P=1 worse while
    staying physical. Verify ``new_M.csv``'s per-P min isn't unexpectedly
    <= 0 before using either.
    """
    df_I = pd.read_csv(os.path.join(DATA_DIR, 'ITD-pol-fit1-line.txt'), header=None, sep=' ')
    df_I.columns = ['W', 'I']
    plot_I_omega(df_I)

    data = pd.read_csv(os.path.join(DATA_DIR, 'All_p_w_m.csv'))

    summary = []
    all_new_M = []
    for fm in sorted(data.FM.unique()):
        df_g = real_mean_M(data, fm)
        n_p = int(df_g.P.nunique())

        # real omega grid for this ensemble, with I(omega) interpolated onto it
        omega = np.sort(df_g['W'].unique())
        I_omega = np.interp(omega, df_I['W'].values, df_I['I'].values)

        gamma = 0.0
        if fit_method == 'gamma':
            alpha, beta, gamma, obj = fit_alpha_beta_gamma(df_g, omega, I_omega, n_p, i=i, j=j)
        elif fit_method == 'monotonic':
            alpha, beta, obj = fit_alpha_beta_monotonic(df_g, omega, I_omega, n_p, p_weight_exp,
                                                         monotonic_penalty, i=i, j=j)
        elif fit_method == 'weighted':
            alpha, beta, obj = fit_alpha_beta_weighted(df_g, omega, I_omega, n_p, p_weight_exp, i=i, j=j)
        else:
            alpha, beta, obj = fit_alpha_beta(df_g, omega, I_omega, n_p, i=i, j=j)

        # per-epoch loss trace for the mi/mj gradient descent
        mi, mj, df_loss = fit_mi_mj(I_omega, alpha, beta, i, j, record_loss=True, omega=omega)
        df_loss.to_csv(os.path.join(DATA_DIR, f'loss_val_FM_{fm}.csv'), index=False)

        # real-vs-estimated comparison uses only the matched (sparse) omega grid
        est = estimate_M(omega, I_omega, n_p, alpha, beta, gamma, i=i, j=j)
        df_cmp = pd.merge(df_g, est, on=['P', 'Wkey'], how='inner', suffixes=('', '_est'))
        df_cmp = df_cmp[['P', 'W', 'Mean_M', 'Est_M']].sort_values(['P', 'W'])

        df_cmp.to_csv(os.path.join(DATA_DIR, f'M_real_vs_est_FM_{fm}.csv'), index=False)

        # new_M is evaluated on the full ITD omega grid (0..20), not just the
        # sparse real grid used for fitting.
        est_full = estimate_M(df_I['W'].values, df_I['I'].values, n_p, alpha, beta, gamma, i=i, j=j)
        new_M = est_full[['P', 'W', 'Est_M']].copy()
        new_M['FM'] = fm
        all_new_M.append(new_M)

        plot_real_vs_synthetic(df_g, est_full, fm, alpha, beta, gamma=gamma)
        plot_real_vs_synthetic(df_g, est_full, fm, alpha, beta, full_range=True, gamma=gamma)
        plot_real_vs_synthetic_combined(df_g, est_full, fm, alpha, beta, gamma=gamma)
        plot_real_vs_synthetic_combined(df_g, est_full, fm, alpha, beta, full_range=True, gamma=gamma)

        n_matched = len(df_cmp)
        mean_abs = np.abs(df_cmp['Mean_M'] - df_cmp['Est_M']).mean()
        summary.append({
            'FM': fm, 'fit_method': fit_method, 'alpha': alpha, 'beta': beta, 'gamma': gamma,
            'objective': obj, 'n_matched': n_matched, 'mean_abs_diff': mean_abs,
        })
        print(f'FM={fm}: method={fit_method} alpha={alpha:.5g}, beta={beta:.5g}, '
              f'gamma={gamma:.5g}, mean|dM|={mean_abs:.5g} over {n_matched} points')

    pd.concat(all_new_M, ignore_index=True).to_csv(os.path.join(DATA_DIR, 'new_M.csv'), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(DATA_DIR, 'optimal_alpha_beta.csv'), index=False)


def _per_p_rel_rms(df_real_g, est):
    """Per-P  RMS(Mean_M - Est_M) / RMS(Mean_M)  on the matched omega grid."""
    m = pd.merge(df_real_g[['P', 'Wkey', 'Mean_M']], est, on=['P', 'Wkey'], how='inner')
    out = {}
    for p in sorted(m.P.unique()):
        t = m[m.P == p]
        denom = np.sqrt(np.mean(t.Mean_M ** 2)) or np.nan
        out[int(p)] = float(np.sqrt(np.mean((t.Mean_M - t.Est_M) ** 2)) / denom)
    return out


def preview_alpha_beta(fm, alpha, beta, gamma=0.0, i=1, j=2, tag='_manual'):
    """Hand-tune (alpha, beta[, gamma]) for one FM and see the effect immediately.

    Skips ``create_combined_data`` and the ``fit_alpha_beta*`` search entirely --
    it just reloads ``All_p_w_m.csv``/``ITD-pol-fit1-line.txt`` (already on disk
    after one run of ``generate_m_omega_from_I_omega``), builds ``estimate_M`` at
    the values you pass in, prints the per-P fit quality, and rewrites
    ``M_real_vs_synthetic*_FM_{fm}{tag}*.pdf`` (``tag`` defaults to ``'_manual'``
    so these never overwrite the pipeline's own ``M_real_vs_synthetic*_FM_{fm}.pdf``
    from ``generate_m_omega_from_I_omega``/``new_M.csv``). Call it repeatedly --
    e.g. from a notebook/REPL -- with different alpha/beta until P=1..4 line up
    well enough; nothing here writes ``new_M.csv`` or ``optimal_alpha_beta.csv``,
    so it's safe to explore without disturbing the pipeline's saved fit.

    Only ``mi``/``mj`` (P = i, j; default 1, 2) are direct functions of
    alpha/beta -- every other P (including P=3, 4) is the closed-form derived
    moment from ``estimate_M``, so alpha/beta move P=3..5 only indirectly
    through mi/I; ``gamma`` (default 0, i.e. off) is the other knob available,
    damping just the derived moments -- see ``estimate_M``'s docstring.

    Returns ``(df_cmp, per_p)``: the matched real-vs-estimated rows (as
    ``generate_m_omega_from_I_omega`` writes to ``M_real_vs_est_FM_{fm}.csv``)
    and a per-P DataFrame of ``mean_abs_diff`` / ``rel_rms_pct`` / ``n`` so you
    can compare candidates numerically, not just by eyeballing the plots.
    """
    df_I = pd.read_csv(os.path.join(DATA_DIR, 'ITD-pol-fit1-line.txt'), header=None, sep=' ')
    df_I.columns = ['W', 'I']

    data = pd.read_csv(os.path.join(DATA_DIR, 'All_p_w_m.csv'))
    df_g = real_mean_M(data, fm)
    n_p = int(df_g.P.nunique())

    omega = np.sort(df_g['W'].unique())
    I_omega = np.interp(omega, df_I['W'].values, df_I['I'].values)

    est = estimate_M(omega, I_omega, n_p, alpha, beta, gamma, i=i, j=j)
    df_cmp = pd.merge(df_g, est, on=['P', 'Wkey'], how='inner', suffixes=('', '_est'))
    df_cmp = df_cmp[['P', 'W', 'Mean_M', 'Est_M']].sort_values(['P', 'W'])

    rel_rms = _per_p_rel_rms(df_g, est)
    rows = []
    for p in sorted(df_cmp.P.unique()):
        d = df_cmp[df_cmp.P == p]
        rows.append({
            'P': int(p),
            'mean_abs_diff': float(np.abs(d['Mean_M'] - d['Est_M']).mean()),
            'rel_rms_pct': 100.0 * rel_rms.get(int(p), np.nan),
            'n': len(d),
        })
    per_p = pd.DataFrame(rows).set_index('P')

    print(f'FM={fm}  alpha={alpha:.5g}  beta={beta:.5g}  gamma={gamma:.5g}')
    print(per_p.to_string(float_format=lambda x: f'{x:.4g}'))

    est_full = estimate_M(df_I['W'].values, df_I['I'].values, n_p, alpha, beta, gamma, i=i, j=j)

    plot_real_vs_synthetic(df_g, est_full, fm, alpha, beta, gamma=gamma, tag=tag)
    plot_real_vs_synthetic(df_g, est_full, fm, alpha, beta, full_range=True, gamma=gamma, tag=tag)
    plot_real_vs_synthetic_combined(df_g, est_full, fm, alpha, beta, gamma=gamma, tag=tag)
    plot_real_vs_synthetic_combined(df_g, est_full, fm, alpha, beta, full_range=True, gamma=gamma, tag=tag)

    return df_cmp, per_p


def diagnose_fit_methods():
    """Compare the baseline fit, option D (re-weighting only) and option A (gamma).

    Prints a per-P relative-RMS table per ensemble and writes
    ``Data/2026/fit_method_comparison.csv``.
    """
    df_I = pd.read_csv(os.path.join(DATA_DIR, 'ITD-pol-fit1-line.txt'), header=None, sep=' ')
    df_I.columns = ['W', 'I']
    data = pd.read_csv(os.path.join(DATA_DIR, 'All_p_w_m.csv'))

    rows = []
    for fm in sorted(data.FM.unique()):
        df_g = real_mean_M(data, fm)
        n_p = int(df_g.P.nunique())
        omega = np.sort(df_g['W'].unique())
        I_omega = np.interp(omega, df_I['W'].values, df_I['I'].values)

        runs = {}
        a, b, _ = fit_alpha_beta(df_g, omega, I_omega, n_p)
        runs['baseline'] = (a, b, 0.0)
        for k in (0.0, 0.5, 1.0):
            a, b, _ = fit_alpha_beta_weighted(df_g, omega, I_omega, n_p, p_weight_exp=k)
            runs[f'D exp={k}'] = (a, b, 0.0)
        a, b, gm, _ = fit_alpha_beta_gamma(df_g, omega, I_omega, n_p)
        runs['A gamma'] = (a, b, gm)

        print(f'\nFM={fm}   per-P relative RMS  (RMS|Δ| / RMS|real|)')
        print(f"  {'method':10s} {'alpha':>8s} {'beta':>6s} {'gamma':>6s}   " +
              '  '.join(f'P{p}' for p in range(1, n_p + 1)))
        for name, (a, b, gm) in runs.items():
            est = estimate_M(omega, I_omega, n_p, a, b, gm)
            rel = _per_p_rel_rms(df_g, est)
            print(f"  {name:10s} {a:8.2f} {b:6.2f} {gm:6.2f}   " +
                  '  '.join(f'{rel.get(p, float("nan")):.2f}' for p in range(1, n_p + 1)))
            rows.append({'FM': fm, 'method': name, 'alpha': a, 'beta': b, 'gamma': gm,
                         **{f'relRMS_P{p}': rel.get(p) for p in range(1, n_p + 1)}})

            est_full = estimate_M(df_I['W'].values, df_I['I'].values, n_p, a, b, gm)
            tag = '_' + name.replace(' ', '').replace('=', '')
            plot_real_vs_synthetic_combined(df_g, est_full, fm, a, b, gamma=gm, tag=tag)

    pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, 'fit_method_comparison.csv'), index=False)


def _fit_std_model(w_real, std_real, cap=True):
    """Linear  std(omega) = slope*omega + intercept.

    ``cap=True`` (default) clips to ``[min(std_real), 3*max(std_real)]`` so a
    long extrapolation can't run away -- used by
    ``find_distribution_of_real_data`` where the full omega grid is only
    ~2-5x the real omega range. ``cap=False`` only floors at a tiny positive
    value (so sigma stays valid) and otherwise lets the line keep growing --
    a genuine linear extrapolation, for ``extrapolate_std_to_new_M`` where the
    full grid reaches ~10x the real omega range and flattening it would
    defeat the point of extrapolating at all.
    """
    slope, intercept = np.polyfit(w_real, std_real, 1)
    lo = float(np.min(std_real)) if cap else 1e-6 * float(np.max(std_real))
    hi = 3.0 * float(np.max(std_real)) if cap else np.inf

    def model(w):
        return np.clip(slope * np.asarray(w, dtype=float) + intercept, lo, hi)

    return model, float(slope), float(intercept)


def find_distribution_of_real_data(omega_grid='full', seed=12345, make_plots=True):
    """Per (FM, P): characterise the Exp-to-Exp scatter of the real data and use
    that same distribution to spread the synthetic mean curve (``new_M.csv``)
    into ``n`` replicas -- ``n`` = the real Exp count for that (FM, P).

    Model (verified against the real data)
    --------------------------------------
    The standardised residual  z = (M_exp(w) - mean(w)) / std(w)  is ~ N(0, 1)
    with the *same* distribution at every omega (skew ~ 0, excess kurtosis ~ 0),
    and  std(w)  grows almost linearly with omega.  The real Exp replicas are
    correlated across omega (rho ~ 0.9).  So per (FM, P):

      * fit  std(w) ~ slope*w + intercept  to the real per-omega std;
      * draw  z_i(w) = sqrt(rho) * u_i + sqrt(1 - rho) * e_i(w)  with
        u_i, e_i(w) ~ N(0, 1)  -> unit variance, cross-omega correlation rho;
      * synthetic replica:  M_i(w) = Est_M(w) + z_i(w) * std_model(w),
        with  M_i(0) = 0  enforced.

    ``omega_grid`` -- ``'full'`` (new_M.csv's 0..20 grid) or ``'real'`` (only the
    measured omega values, interpolating Est_M onto them).

    Writes ``Data/2026/real_distribution_params.csv`` and
    ``Data/2026/Synthetic_M_replicas.csv`` (columns FM, P, W, Exp, M).
    """
    data = pd.read_csv(os.path.join(DATA_DIR, 'All_p_w_m.csv'))
    new_M = pd.read_csv(os.path.join(DATA_DIR, 'new_M.csv'))

    params, replicas, checks = [], [], []
    for (fm, p), t in data[data.W > 0].groupby(['FM', 'P']):
        piv = t.pivot_table(index='Exp', columns='W', values='M').dropna(axis=1, how='any')
        n = int(piv.shape[0])
        w_real = piv.columns.values.astype(float)
        mean_w = piv.mean().to_numpy()
        std_w = piv.std().to_numpy()

        z = ((piv.to_numpy() - mean_w) / std_w).ravel()
        z = z[np.isfinite(z)]

        corr = np.corrcoef(piv.to_numpy().T)
        rho = float(np.clip(np.nanmean(corr[np.triu_indices_from(corr, 1)]), 0.0, 0.999))

        std_model, slope, intercept = _fit_std_model(w_real, std_w)

        params.append({
            'FM': fm, 'P': p, 'n_exp': n, 'std_slope': slope, 'std_intercept': intercept,
            'z_mean': float(z.mean()), 'z_std': float(z.std()),
            'z_skew': float(stats.skew(z)), 'z_excess_kurtosis': float(stats.kurtosis(z)),
            'cross_w_corr': rho,
        })

        # synthetic mean curve on the requested omega grid
        s = new_M[(new_M.FM == fm) & (new_M.P == p)].sort_values('W')
        if omega_grid == 'real':
            w = np.concatenate(([0.0], w_real))
        else:
            w = s['W'].to_numpy(dtype=float)
        est = np.interp(w, s['W'].to_numpy(dtype=float), s['Est_M'].to_numpy(dtype=float))

        rng = np.random.default_rng(seed + int(fm) * 100 + int(p))
        u = rng.standard_normal((n, 1))
        e = rng.standard_normal((n, w.size))
        zmat = np.sqrt(rho) * u + np.sqrt(1.0 - rho) * e
        M = est[None, :] + zmat * std_model(w)[None, :]
        M[:, w == 0.0] = 0.0

        replicas.append(pd.DataFrame({
            'FM': fm, 'P': p,
            'W': np.tile(w, n),
            'Exp': np.repeat(np.arange(n), w.size),
            'M': M.ravel(),
        }))

        # book-keeping: does the synthetic scatter reproduce the real one?
        for wi, sr in zip(w_real, std_w):
            checks.append({'FM': fm, 'P': p, 'W': wi, 'std_real': sr,
                           'std_model': float(std_model(wi))})

    df_params = pd.DataFrame(params)
    df_params.to_csv(os.path.join(DATA_DIR, 'real_distribution_params.csv'), index=False)
    pd.concat(replicas, ignore_index=True).to_csv(
        os.path.join(DATA_DIR, 'Synthetic_M_replicas.csv'), index=False)

    print('\nreal-data distribution per (FM, P)  [z ~ N(0,1) in all cases]')
    print(df_params.round(4).to_string(index=False))

    if make_plots:
        df_chk = pd.DataFrame(checks)
        for fm, g in df_chk.groupby('FM'):
            all_p = sorted(g.P.unique())
            fig, axes = plt.subplots(1, len(all_p), figsize=(5 * len(all_p), 4.5),
                                     dpi=300, squeeze=False)
            for ax, pp in zip(axes.flat, all_p):
                tp = g[g.P == pp].sort_values('W')
                ax.plot(tp.W, tp.std_real, '^', label='real std')
                ax.plot(tp.W, tp.std_model, '-', label='model std')
                ax.set_title(fr'$p_z$={pp}')
                ax.set_xlabel(r'$\omega$')
                ax.set_ylabel(r'std$[\,M(\omega)\,]$')
                ax.legend(fontsize=10)
            fig.suptitle(f'FM={fm}   real vs modelled Exp-scatter')
            fig.tight_layout()
            fig.savefig(os.path.join(IMG_DIR, f'real_distribution_FM_{fm}.pdf'),
                        format='pdf', dpi=300)
            plt.close(fig)

    return df_params


# Candidate shapes to test the real per-(FM,P,W) M distribution against.
# norm/logistic/laplace: 2 params (progressively heavier tails); t: 3 params,
# heavier tails than normal but symmetric; skewnorm: 3 params, allows skew.
_CANDIDATE_DISTS = {
    'norm': stats.norm,
    't': stats.t,
    'logistic': stats.logistic,
    'laplace': stats.laplace,
    'skewnorm': stats.skewnorm,
}


def _fit_and_score(x, dist):
    """MLE-fit ``dist`` to ``x``; return AIC/BIC (model-comparison, penalises
    extra parameters) and a KS statistic (absolute goodness-of-fit)."""
    try:
        params = dist.fit(x)
        loglik = float(np.sum(dist.logpdf(x, *params)))
        if not np.isfinite(loglik):
            raise ValueError('non-finite log-likelihood')
        k = len(params)
        n = len(x)
        ks_stat, ks_p = stats.kstest(x, dist.cdf, args=params)
        return {'aic': 2 * k - 2 * loglik, 'bic': k * np.log(n) - 2 * loglik,
                'ks_stat': float(ks_stat), 'ks_p': float(ks_p)}
    except Exception:
        return {'aic': np.inf, 'bic': np.inf, 'ks_stat': np.nan, 'ks_p': np.nan}


def _plot_distribution_example(data, fm, p, w, names, tag):
    x = data[(data.FM == fm) & (data.P == p) & np.isclose(data.W, w)]['M'].to_numpy()
    plt.figure(dpi=300)
    counts, bins, _ = plt.hist(x, bins=40, density=True, alpha=0.35, color='gray', label='real M')
    grid = np.linspace(bins[0], bins[-1], 400)
    for name in names:
        params = _CANDIDATE_DISTS[name].fit(x)
        plt.plot(grid, _CANDIDATE_DISTS[name].pdf(grid, *params), lw=2, label=name)
    plt.xlabel('M')
    plt.ylabel('density')
    plt.title(fr'FM={fm} $p_z$={p} $\omega$={w:.3g}  (n={len(x)})')
    plt.legend(fontsize=11)
    plt.savefig(os.path.join(IMG_DIR, f'M_distribution_example_{tag}.pdf'), format='pdf', dpi=300)
    plt.close()


def check_M_distribution(min_n=30, make_plots=True):
    """For every (FM, P, W) with real data, fit each of ``_CANDIDATE_DISTS`` to
    the n Exp values of M and see which shape wins most often.

    Model selection uses AIC (lower is better; penalises the 3-parameter
    distributions for the extra flexibility) so a distribution only "wins" by
    genuinely fitting better, not by having more free parameters. The KS
    statistic is reported too as an absolute (sample-size-independent) measure
    of fit quality -- unlike its p-value, which is not usable here: with
    n ~ 1000 even a tiny, physically irrelevant deviation from a candidate
    shape drives the p-value to ~0 for every distribution.

    Writes ``Data/2026/distribution_fit_comparison.csv`` (one row per
    (FM,P,W)) and prints the win-count / mean-rank summary.
    """
    data = pd.read_csv(os.path.join(DATA_DIR, 'All_p_w_m.csv'))
    data = data[data.W > 0]

    names = list(_CANDIDATE_DISTS)
    rows = []
    for (fm, p, w), t in data.groupby(['FM', 'P', 'W']):
        x = t['M'].to_numpy(dtype=float)
        if len(x) < min_n:
            continue
        row = {'FM': fm, 'P': p, 'W': w, 'n': len(x)}
        for name in names:
            s = _fit_and_score(x, _CANDIDATE_DISTS[name])
            row[f'{name}_aic'] = s['aic']
            row[f'{name}_bic'] = s['bic']
            row[f'{name}_ks_stat'] = s['ks_stat']
        aics = {name: row[f'{name}_aic'] for name in names}
        row['best_aic_dist'] = min(aics, key=aics.get)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, 'distribution_fit_comparison.csv'), index=False)

    n_groups = len(df)
    print(f'\nDistribution fit over {n_groups} (FM,P,W) groups '
          f'(n_Exp per group: {df.n.min()}-{df.n.max()})')

    print('\nwin count (lowest AIC):')
    print(df['best_aic_dist'].value_counts().reindex(names, fill_value=0).to_string())

    print('\nmean AIC - mean(norm AIC)  (negative = fits better than normal on average):')
    for name in names:
        if name == 'norm':
            continue
        d = df[f'{name}_aic'] - df['norm_aic']
        d = d[np.isfinite(d)]
        print(f'  {name:9s} {d.mean():+9.3f}   (better in {(d < 0).mean():.0%} of groups)')

    print('\nmean KS statistic (lower = better absolute fit):')
    print(df[[f'{name}_ks_stat' for name in names]].mean()
          .rename(lambda c: c.replace('_ks_stat', '')).to_string())

    rank = df[[f'{name}_aic' for name in names]].rank(axis=1)
    rank.columns = names
    print('\nmean AIC rank (1=best):')
    print(rank.mean().sort_values().round(2).to_string())

    if make_plots:
        # a typical group (largest n) and the group where a non-normal shape
        # helped most (most negative skewnorm-vs-normal AIC gap)
        typical = df.loc[df.n.idxmax()]
        skewed = df.loc[(df.skewnorm_aic - df.norm_aic).idxmin()]
        _plot_distribution_example(data, typical.FM, typical.P, typical.W, names, 'typical')
        _plot_distribution_example(data, skewed.FM, skewed.P, skewed.W, names, 'most_skewed')

    return df


def _real_groups(data=None):
    """(FM, P, W) real-data groups, W>0, as {(fm,p,w): M-array}."""
    if data is None:
        data = pd.read_csv(os.path.join(DATA_DIR, 'All_p_w_m.csv'))
    data = data[data.W > 0]
    return {k: t['M'].to_numpy(dtype=float) for k, t in data.groupby(['FM', 'P', 'W'])}


def _draw_mean_vs_median(ax, groups, title='mean vs median'):
    """Original diagnostic: mean vs median of M per (FM,P,W).

    Kept for reference, but this only tests *symmetry* -- mean==median holds
    for any symmetric unimodal distribution (normal, logistic, Laplace,
    Student-t, uniform...), and with n~1000 the sample mean/median are both so
    precise that near-perfect agreement is expected regardless of shape. It
    cannot tell a normal apart from a heavier- or lighter-tailed symmetric
    alternative -- see ``_draw_qq_normal`` / ``plot_skew_kurtosis`` instead.
    """
    mean = np.array([x.mean() for x in groups.values()])
    median = np.array([np.median(x) for x in groups.values()])

    lo, hi = min(mean.min(), median.min()), max(mean.max(), median.max())
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label='y = x')
    ax.scatter(mean, median, s=14, alpha=0.6)
    ax.set_xlabel('mean(M)')
    ax.set_ylabel('median(M)')
    ax.set_title(title)
    ax.legend()


def _draw_qq_normal(ax, groups, title='qq-normal'):
    """Q-Q plot of the pooled standardised residuals z=(M-mean)/std against the
    theoretical normal quantiles -- the standard visual test for normality.
    Unlike mean-vs-median it is sensitive to the whole shape (skew AND tail
    weight/kurtosis), and pooling every (FM,P,W) group onto one plot (all
    groups share the same z~N(0,1) shape, only the scale differs) gives one
    figure with ~10^5 points instead of ~100 single-number summaries.
    """
    z = np.concatenate([(x - x.mean()) / x.std(ddof=1) for x in groups.values()])

    stats.probplot(z, dist='norm', plot=ax)
    line = ax.get_lines()
    line[0].set_markersize(2)
    line[0].set_alpha(0.3)
    line[0].set_rasterized(True)  # ~1e5 points -> keep the PDF small
    ax.set_title(title)


def plot_mean_vs_median(data=None):
    plt.figure(dpi=300)
    _draw_mean_vs_median(plt.gca(), _real_groups(data), title='mean vs median of M, per (FM, P, W)')
    plt.savefig(os.path.join(IMG_DIR, 'mean_vs_median.pdf'), format='pdf', dpi=300)
    plt.close()


def plot_qq_normal(data=None):
    groups = _real_groups(data)
    plt.figure(dpi=300)
    _draw_qq_normal(plt.gca(), groups,
                    title=f'Normal Q-Q plot, standardised M  '
                          f'({sum(len(x) for x in groups.values())} points pooled over all (FM,P,W))')
    plt.savefig(os.path.join(IMG_DIR, 'qq_normal_pooled.pdf'), format='pdf', dpi=300)
    plt.close()


def plot_mean_median_and_qq(data=None):
    """Mean-vs-median and the normal Q-Q plot side by side in one figure."""
    groups = _real_groups(data)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)
    _draw_mean_vs_median(ax1, groups, title='mean vs median')
    _draw_qq_normal(ax2, groups, title='qq-normal')
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'mean_median_and_qq.pdf'), format='pdf', dpi=300)
    plt.close(fig)


def plot_skew_kurtosis(data=None):
    """Scatter of (skewness, excess kurtosis) per (FM,P,W) -- normal sits at the
    origin. Reference markers show where the other candidates from
    ``check_M_distribution`` would sit (all symmetric -> skew=0, so this is the
    kurtosis axis that actually separates them; skewness is on the other axis
    to also catch any asymmetry mean-vs-median would miss).
    """
    groups = _real_groups(data)
    rows = [{'skew': stats.skew(x), 'kurt': stats.kurtosis(x)} for x in groups.values()]
    df = pd.DataFrame(rows)

    plt.figure(dpi=300)
    plt.scatter(df['skew'], df['kurt'], s=14, alpha=0.6, label='(FM,P,W) groups', zorder=3)
    plt.plot(0, 0, 'k*', ms=20, label='normal', zorder=4)
    for label, kurt in [('logistic', 1.2), ('laplace', 3.0), ('uniform', -1.2)]:
        plt.plot(0, kurt, 'x', ms=10, mew=2, label=label, zorder=4)
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.xlabel('skewness')
    plt.ylabel('excess kurtosis')
    plt.title('skewness vs excess kurtosis, per (FM, P, W)')
    plt.legend(fontsize=10)
    plt.savefig(os.path.join(IMG_DIR, 'skew_vs_kurtosis.pdf'), format='pdf', dpi=300)
    plt.close()

def _poly_r2(x, y, deg):
    """R^2 of a degree-``deg`` polynomial fit to (x, y); (nan, None) if too few points."""
    if len(x) <= deg:
        return np.nan, None
    coeffs = np.polyfit(x, y, deg)
    yhat = np.polyval(coeffs, x)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(r2), coeffs


def check_linearity(data=None, quad_r2_gain=0.02):
    """Is  mean(M) vs omega  and  std(M) vs omega  linear or non-linear, per (FM, P)?

    Fits a line (1 param slope + intercept) and a quadratic to each curve and
    compares R^2: if the quadratic's R^2 only improves on the line's by less
    than ``quad_r2_gain`` (2% by default), the curvature isn't buying
    anything -- call it linear; otherwise non-linear. This is descriptive
    (each curve only has 5-10 real omega points), not a formal hypothesis
    test, but is consistent across all (FM, P).

    Writes ``Data/2026/linearity_check.csv``.
    """
    if data is None:
        data = pd.read_csv(os.path.join(DATA_DIR, 'All_p_w_m.csv'))

    rows = []
    for fm in sorted(data.FM.unique()):
        df_g = real_mean_M(data, fm)
        for p in sorted(df_g.P.unique()):
            t = df_g[df_g.P == p].sort_values('W')
            w = t['W'].to_numpy(dtype=float)
            for quantity, y in (('mean', t['Mean_M'].to_numpy(dtype=float)),
                                ('std', t['Std_M'].to_numpy(dtype=float))):
                r2_lin, _ = _poly_r2(w, y, 1)
                r2_quad, _ = _poly_r2(w, y, 2)
                gain = r2_quad - r2_lin if np.isfinite(r2_lin) and np.isfinite(r2_quad) else np.nan
                verdict = 'linear' if (np.isfinite(gain) and gain < quad_r2_gain) else 'non-linear'
                rows.append({'FM': fm, 'P': p, 'quantity': quantity, 'n': len(w),
                             'r2_linear': r2_lin, 'r2_quadratic': r2_quad,
                             'r2_gain': gain, 'verdict': verdict})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, 'linearity_check.csv'), index=False)

    print('\nlinear vs non-linear trend vs omega, per (FM, P):')
    print(df.pivot_table(index=['FM', 'P'], columns='quantity',
                         values='verdict', aggfunc='first').to_string())

    print('\nfraction of (FM,P) groups classified linear, and mean R^2 (linear fit):')
    summary = df.groupby('quantity').agg(
        frac_linear=('verdict', lambda s: (s == 'linear').mean()),
        mean_r2_linear=('r2_linear', 'mean'))
    print(summary.round(3).to_string())

    return df


def _plot_by_p(data, ycol, ylabel, out_prefix, show_linear_fit):
    """Shared driver for the mean(M)/std(M) vs omega, grouped-by-P plots."""
    for fm in sorted(data.FM.unique()):
        df_g = real_mean_M(data, fm)
        plt.figure(dpi=300)
        for p in sorted(df_g.P.unique()):
            t = df_g[df_g.P == p].sort_values('W')
            w, y = t['W'].to_numpy(dtype=float), t[ycol].to_numpy(dtype=float)
            line, = plt.plot(w, y, 'o', label=fr"$p_z$={p}")
            if show_linear_fit:
                r2, coeffs = _poly_r2(w, y, 1)
                if coeffs is not None:
                    plt.plot(w, np.polyval(coeffs, w), '--', color=line.get_color(), lw=1,
                             label=fr"$p_z$={p} linear fit ($R^2$={r2:.3f})")
        plt.xlabel(r'$\omega$')
        plt.ylabel(ylabel)
        plt.title(f'FM={fm}')
        plt.legend(fontsize=10, ncol=2)
        plt.savefig(os.path.join(IMG_DIR, f'{out_prefix}_FM_{fm}.pdf'), format='pdf', dpi=300)
        plt.close()


def plot_mean_vs_w(data=None, show_linear_fit=True):
    """For each FM: mean(M) vs omega, one curve per P (+ linear-fit overlay)."""
    if data is None:
        data = pd.read_csv(os.path.join(DATA_DIR, 'All_p_w_m.csv'))
    _plot_by_p(data, 'Mean_M', 'mean(M)', 'Mean_vs_W', show_linear_fit)


def plot_std_vs_w(data=None, show_linear_fit=True):
    """For each FM: std(M) vs omega, one curve per P (+ linear-fit overlay)."""
    if data is None:
        data = pd.read_csv(os.path.join(DATA_DIR, 'All_p_w_m.csv'))
    _plot_by_p(data, 'Std_M', 'std(M)', 'Std_vs_W', show_linear_fit)


def extrapolate_std_to_new_M(data=None, new_M=None):
    """Step 1-2: extrapolate std(omega) onto new_M.csv's full omega grid.

    Per (FM, P), fits the linear ``std(omega) = slope*omega + intercept``
    model (``_fit_std_model``, clipped to the real std range) to the real
    data, then evaluates it at every omega in ``new_M.csv`` (0..20, includes
    omega beyond the real range -- this is the extrapolation).

    Writes ``Data/2026/new_M_S.csv`` with columns FM, W, P, Est_M, Est_S.
    """
    if data is None:
        data = pd.read_csv(os.path.join(DATA_DIR, 'All_p_w_m.csv'))
    if new_M is None:
        new_M = pd.read_csv(os.path.join(DATA_DIR, 'new_M.csv'))

    data = data[data.W > 0]
    parts = []
    for (fm, p), t in data.groupby(['FM', 'P']):
        std_w = t.groupby('W')['M'].std()
        model, _, _ = _fit_std_model(std_w.index.to_numpy(dtype=float), std_w.to_numpy(), cap=False)

        g = new_M[(new_M.FM == fm) & (new_M.P == p)].copy()
        g['Est_S'] = model(g['W'].to_numpy(dtype=float))
        parts.append(g)

    new_M_S = pd.concat(parts, ignore_index=True)[['FM', 'W', 'P', 'Est_M', 'Est_S']]
    new_M_S = new_M_S.sort_values(['FM', 'P', 'W']).reset_index(drop=True)
    new_M_S.to_csv(os.path.join(DATA_DIR, 'new_M_S.csv'), index=False)
    return new_M_S


def max_n_exp_per_fm(data=None):
    """Step 3: N_fm = the largest real Exp count over P, for each FM."""
    if data is None:
        data = pd.read_csv(os.path.join(DATA_DIR, 'All_p_w_m.csv'))
    n_per_fm_p = data.groupby(['FM', 'P'])['Exp'].nunique()
    return n_per_fm_p.groupby(level='FM').max().to_dict()


def generate_synthetic_exp_data(new_M_S=None, N_fm=None, seed=12345):
    """Steps 4-5: for each (FM, P, W) draw N_fm[FM] i.i.d. replicas of M from
    Normal(Est_M, Est_S) truncated to M >= 0 -- N_fm Exp replicas at every
    omega on the full 0..20 grid.

    M (RpITD) is a physically non-negative magnitude, so an unbounded
    Normal is wrong here: wherever Est_S is comparable to Est_M (the W=0
    anchor rows, where Est_M==0 exactly, and high-P/large-W rows near the
    noise floor) an unbounded draw would land negative on a large fraction
    of replicas. Truncating at 0 keeps every replica physical without
    clipping (which would pile up mass at exactly 0 and distort the
    variance) by resampling that side of the Gaussian via its CDF.

    Writes ``Data/2026/generated_Synthetic_exp_data.csv`` with columns
    FM, W, P, Est_M, Est_S, Exp, M.
    """
    if new_M_S is None:
        new_M_S = pd.read_csv(os.path.join(DATA_DIR, 'new_M_S.csv'))
    if N_fm is None:
        N_fm = max_n_exp_per_fm()

    rng = np.random.default_rng(seed)
    parts = []
    for fm, g in new_M_S.groupby('FM'):
        g = g.reset_index(drop=True)   # so positions below index straight into `samples`
        n = int(N_fm[fm])
        mu = g['Est_M'].to_numpy(dtype=float)
        sigma = g['Est_S'].to_numpy(dtype=float)

        # a = (lower_bound - mu) / sigma is truncnorm's lower shape param in
        # standardized units; upper bound is +inf. Guard sigma<=0 (none seen
        # in practice, but _fit_std_model isn't guaranteed to stay positive)
        # by substituting a dummy scale and overwriting those rows with mu.
        safe_sigma = np.where(sigma > 0, sigma, 1.0)
        a = (0.0 - mu) / safe_sigma
        samples = stats.truncnorm.rvs(a[:, None], np.inf, loc=mu[:, None],
                                       scale=safe_sigma[:, None],
                                       size=(len(g), n), random_state=rng)
        samples = np.where(sigma[:, None] > 0, samples, mu[:, None])

        # Smooth each replica's own M(omega) trajectory along the W grid --
        # NOT samples.ravel(), which interleaves Exp fastest and would smooth
        # across unrelated i.i.d. replicas of the same (P,W) point (and bleed
        # across (P,W) block boundaries) instead of along omega. `g` is
        # sorted P-then-W (new_M_S's sort order), so each P's rows are
        # contiguous; smoothing axis=0 within a block smooths each column
        # (one Exp replica) down that P's W-ordered curve.
        smoothed = np.empty_like(samples)
        for _, idx in g.groupby('P').indices.items():
            smoothed[idx, :] = np.apply_along_axis(get_smooth_curve, 0, samples[idx, :])

        parts.append(pd.DataFrame({
            'FM': np.repeat(g['FM'].to_numpy(), n),
            'W': np.repeat(g['W'].to_numpy(), n),
            'P': np.repeat(g['P'].to_numpy(), n),
            'Est_M': np.repeat(mu, n),
            'Est_S': np.repeat(sigma, n),
            'Exp': np.tile(np.arange(n), len(g)),
            'M': smoothed.ravel(),
        }))

    out = pd.concat(parts, ignore_index=True)
    out.loc[out.W==0.0, 'M'] = 0.0
    out.to_csv(os.path.join(DATA_DIR, 'generated_Synthetic_exp_data.csv'), index=False)
    return out


def generate_m_omega_from_distribution():
    """Full pipeline: new_M.csv's (FM,P,W)->Est_M, plus a linear-model
    extrapolation of std(omega), used as (mu, sigma) to draw N_fm synthetic
    Exp replicas of M per (FM,P,W). See ``extrapolate_std_to_new_M`` /
    ``generate_synthetic_exp_data`` docstrings for the two output files.
    """
    data = pd.read_csv(os.path.join(DATA_DIR, 'All_p_w_m.csv'))
    new_M_S = extrapolate_std_to_new_M(data)
    N_fm = max_n_exp_per_fm(data)
    print('N_fm (max real Exp count per FM):', N_fm)
    generate_synthetic_exp_data(new_M_S, N_fm)


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Default (no arguments): run the full synthetic-data pipeline '
                    '(fit alpha/beta per FM and generate everything downstream of it). '
                    'Pass --fm together with --alpha/--beta to instead preview a '
                    'hand-picked fit for that one FM via preview_alpha_beta, without '
                    'touching new_M.csv or optimal_alpha_beta.csv.')
    parser.add_argument('--fm', type=int, default=None,
                        help='Ensemble id (e.g. 9, 12, 15). Requires --alpha and --beta.')
    parser.add_argument('--alpha', type=float, default=None, help='alpha to preview for --fm.')
    parser.add_argument('--beta', type=float, default=None, help='beta to preview for --fm.')
    parser.add_argument('--gamma', type=float, default=0.0,
                        help='Optional high-P damping term (see estimate_M). Default 0 (off).')
    parser.add_argument('--i', type=int, default=1, help='Lower momentum of the directly-fit pair.')
    parser.add_argument('--j', type=int, default=2, help='Higher momentum of the directly-fit pair.')
    parser.add_argument('--tag', default='_manual',
                        help="Suffix for the preview plot filenames, so they don't overwrite "
                             "the pipeline's own M_real_vs_synthetic*_FM_{fm}.pdf.")
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    if args.fm is not None or args.alpha is not None or args.beta is not None:
        if args.fm is None or args.alpha is None or args.beta is None:
            raise SystemExit('--fm, --alpha and --beta must all be given together.')
        preview_alpha_beta(args.fm, args.alpha, args.beta, gamma=args.gamma,
                           i=args.i, j=args.j, tag=args.tag)
    else:
        if not os.path.exists(os.path.join(DATA_DIR, 'All_p_w_m.csv')):
            data = create_combined_data()
            plot_p_w_m_curves(data)

        # baseline (gamma=0, so new_M.csv's derived moments stay exactly
        # cross-pair consistent) -- see generate_m_omega_from_I_omega's docstring
        # for why 'weighted'/'monotonic' (meant to prioritise higher P) aren't
        # the default: both produce a degenerate (M<=0 everywhere) P=1 curve for
        # FM=12/15 here.
        generate_m_omega_from_I_omega(fit_method='baseline')
        diagnose_fit_methods()
        find_distribution_of_real_data()
        check_M_distribution()
        #plot_mean_vs_median()
        #plot_qq_normal()
        #plot_skew_kurtosis()
        check_linearity()
        plot_mean_vs_w()
        plot_std_vs_w()
        plot_mean_median_and_qq()
        generate_m_omega_from_distribution()
