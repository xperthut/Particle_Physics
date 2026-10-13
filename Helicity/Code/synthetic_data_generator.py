"""Generate synthetic M(omega, P) replicas per lattice ensemble (``FM`` = 9 / 12 / 15).

Stages run by ``__main__``:

1. ``create_combined_data`` -- concatenate the raw per-momentum RpITD files
   into ``Data/2026/All_p_w_m.csv`` (with a single omega=0 anchor per (FM, P)),
   keeping only the ``N_OMEGA_PER_FM[fm]`` smallest omega values per (FM, P).

2. ``generate_I_omega`` -- per FM, a weighted least-squares fit of the real
   per-(P, omega) mean M to
       M(omega, P) = A * omega * exp(-slope * omega) * P**(-gamma) * exp(c * z2),
       z2 = min((omega / P)**2, z2_max),
   giving I(omega) = A * omega * exp(-slope * omega) (scaled) on omega = 0..20,
   written to ``rpitd/ITD.csv``.

3. ``generate_m_omega_from_I_omega`` -- per FM, derive M(omega, P) from I(omega)
   so that every momentum pair reconstructs the same I:
     a. fit mi = M(omega, P=i), mj = M(omega, P=j) (pair ``FIT_PAIR``) by
        gradient descent on  I = (j^2 * mj - i^2 * mi) / (j^2 - i^2),
        seeded from ``mi = alpha * I``, ``mj = beta * I`` (``abg_config``);
     b. every other P from the closed form  m_k = ((k^2 - i^2) * I + i^2 * mi) / k^2.
   Writes ``new_M.csv`` (omega = 0..20).

4. Diagnostics of the real data (distribution shape, linearity of mean/std
   vs omega) -- CSVs and plots only, nothing downstream reads them.

5. ``generate_m_omega_from_distribution`` -- draw synthetic Exp replicas of M
   around ``new_M.csv`` with a linear std(omega) model, written to
   ``generated_Synthetic_exp_data.csv`` (read by prepare_train_test_data.py).

``--fm/--alpha/--beta`` instead previews hand-picked alpha/beta for one FM
(``preview_alpha_beta``) without touching the pipeline outputs.

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
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

mpl.rcParams['figure.figsize'] = (12, 10)
mpl.rcParams.update({'font.size': 22})

SMOOTH_WINDOW = 60       # Savitzky-Golay window in grid points (0.1 each)
SMOOTH_POLYORDER = 3

PLOT_DPI = 330           # all figures are saved as PNG at this resolution

VERSION = 2026
DATA_DIR = f'../Data/{VERSION}/'
IMG_DIR = f'../Data/{VERSION}/Images/'
# fitted I(omega) lines the synthetic M is built from, one per FM (csv: W, FM, I(W))
ITD_FILE = os.path.join(DATA_DIR, 'rpitd', 'ITD.csv')

# Hand-picked (see preview_alpha_beta) seeds for fit_mi_mj, per FM.
abg_config = {
    '09': {'alpha': 7.5, 'beta': 0.0, 'gamma': 0.0},
    '12': {'alpha': 6.5, 'beta': 0.0, 'gamma': 0.0},
    '15': {'alpha': 5.5, 'beta': 0.0, 'gamma': 0.0},
}

# Number of real omega values kept per (FM, P): the smallest ones, i.e. Wilson-line
# displacements z = 1..N (omega = p*z*const). Per the domain scientist, the larger-z
# points are not used. Applied in create_combined_data, so every consumer of
# All_p_w_m.csv sees only the kept points.
N_OMEGA_PER_FM = {'09': 5, '12': 4, '15': 3}

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

# I(omega) = B*mj + A*mi for the directly-fit momentum pair (i, j), j>i>=2,
# with A = -i^2/(j^2-i^2), B = j^2/(j^2-i^2). FIT_PAIR is this repo's default
# pair everywhere below. Per the domain scientist, P=1 is never used for the
# M estimation, so i >= 2 (P=1's M is derived from the pair's closed form).
FIT_PAIR = (2, 3)

def _check_pair(i, j):
    if not 1 <= i < j:
        raise ValueError(f'momentum pair (i={i}, j={j}) must satisfy 2 <= i < j -- '
                         'P=1 is not used for M estimation.')


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
            # keep only the N_OMEGA_PER_FM[fm] smallest omega values for this P
            kept_w = np.sort(df['W'].unique())[:N_OMEGA_PER_FM[fm]]
            df = df[df['W'].isin(kept_w)].copy()
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


def plot_I_omega(df_I):
    plt.figure(dpi=PLOT_DPI)
    for fm in sorted(df_I.FM.unique()):
        plt.plot(df_I[df_I.FM==fm].W, df_I[df_I.FM==fm].I, '-o', label=f'FM={fm / 100:.2f}')
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\Delta \mathcal{I}(\omega)$')
    plt.legend()
    plt.savefig(os.path.join(IMG_DIR, 'I(w).png'), format='png', dpi=PLOT_DPI)
    plt.close()


def load_itd():
    """Read ``ITD_FILE`` (written by ``generate_I_omega``) as columns ``FM, W, I``."""
    df_I = pd.read_csv(ITD_FILE)
    df_I.columns = ['FM', 'W', 'I']
    df_I['FM'] = df_I['FM'].astype(int)
    return df_I.sort_values(['W']).reset_index(drop=True)


def itd_for_fm(df_I, fm):
    """``(W, I)`` arrays of the I(omega) line for ensemble ``fm`` (e.g. 9 or '09')."""
    g = df_I[df_I.FM == int(fm)]
    if g.empty:
        raise ValueError(f'{ITD_FILE} has no I(omega) rows for FM={fm} '
                         f'(available: {sorted(df_I.FM.unique())}).')
    return g['W'].to_numpy(), g['I'].to_numpy()


def fit_mi_mj(I_omega, alpha, beta, i=FIT_PAIR[0], j=FIT_PAIR[1], record_loss=False, omega=None):
    """Gradient-descent fit of ``mi``, ``mj`` -- the directly-fit momentum
    pair, ``j>i>=2``, default ``FIT_PAIR`` -- to
    ``I = A*mi + B*mj`` for each omega, with
    ``A = -i^2/(j^2-i^2)``, ``B = j^2/(j^2-i^2)`` (see ``_ab_for_pair``).

    ``I_omega`` is a 1-D array of I values.  The seed is ``mi = alpha * I``,
    ``mj = beta * I``.  All omega points are optimised together (they are
    independent).  Returns ``(mi, mj)`` arrays; if ``record_loss`` also returns
    a long-form DataFrame with per-epoch loss (``omega`` labels its rows and
    must be the grid ``I_omega`` is sampled on).

    Both seeds scale with ``I`` so the whole trajectory passes through (0, 0)
    as ``I -> 0``; a flat ``mj = beta`` seed would leave a visible jump next
    to ``estimate_M``'s ``M(omega=0) = 0`` anchor given the 100-epoch budget.
    """
    _check_pair(i, j)
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


def estimate_M(omega, I_omega, n_p, alpha, beta, gamma=0.0, i=FIT_PAIR[0], j=FIT_PAIR[1]):
    """Long-form generated M: columns ``W``, ``P`` (1..n_p), ``Est_M``, ``Wkey``.

    ``i, j`` (default ``FIT_PAIR``, i >= 2) -- the directly-fit momentum pair:
    ``mi``, ``mj`` come from ``fit_mi_mj``; every other P's moment (including
    P=1, which is never fit directly) is the closed form
    ``m_k = ((k^2-i^2)*I + i^2*mi) / k^2``, the algebraic inverse of
    ``I = (k^2*m_k - i^2*mi)/(k^2-i^2)`` for *every* k -- so with gamma=0 any
    two P's reconstruct the same I(omega), a physical requirement (I(omega)
    must not depend on which momentum pair estimated it).

    ``gamma`` -- damping of the derived (P not in {i, j}) moments (option A).
    The closed form tends to ``I(omega)`` as k grows past ``j``, which
    over-predicts the real data at high P.  ``gamma`` multiplies each derived
    moment above ``j`` by ``exp(-gamma * (k - j))`` so the tail can be pulled
    down without touching the directly-fit ``mi``, ``mj`` (moments below ``j``,
    e.g. P=1, are left undamped).  ``gamma=0`` is the original ansatz.
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
        cols[k] = (((k ** 2 - i ** 2) * y + (i ** 2) * mi) / (k ** 2)) * np.exp(-gamma * max(k - j, 0))

    df = pd.DataFrame(cols).melt(id_vars='W', var_name='P', value_name='Est_M')
    df['P'] = df['P'].astype(int)
    df['Wkey'] = _w_key(df['W'])
    return df


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

    plt.figure(dpi=PLOT_DPI)
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
    plt.savefig(os.path.join(IMG_DIR, f'M_real_vs_synthetic_combined_FM_{fm}{tag}{suffix}.png'),
                format='png', dpi=PLOT_DPI)
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
                             dpi=PLOT_DPI, squeeze=False)

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
    fig.savefig(os.path.join(IMG_DIR, f'M_real_vs_synthetic_FM_{fm}{tag}{suffix}.png'),
                format='png', dpi=PLOT_DPI)
    plt.close(fig)


def generate_m_omega_from_I_omega(i=FIT_PAIR[0], j=FIT_PAIR[1]):
    """Write the generated M(omega) tables/plots per FM, using ``abg_config``.

    ``i, j`` (default ``FIT_PAIR``, ``j>i>=2``) -- the directly-fit momentum pair for
    every FM (see ``estimate_M``); every other P's moment is derived from it.

    Every FM in ``All_p_w_m.csv`` must have an ``abg_config`` entry (keyed by
    the zero-padded FM string, e.g. ``'09'``), typically hand-picked with
    ``preview_alpha_beta``; a missing one raises ``ValueError``.

    With ``gamma=0`` (``abg_config``'s default), every derived (P not in
    {i, j}) moment is exactly m_k = ((k^2-i^2)*I + i^2*mi) / k^2, the algebraic
    inverse of I = (k^2*m_k - i^2*mi)/(k^2-i^2) for *every* k -- so any two P's
    reconstruct the *same* I(omega) (same value, same sign as the fitted
    curve), which is a physical requirement (I(omega) must not depend on
    which momentum pair estimated it). A nonzero ``gamma`` damps the derived
    moments to better match the real per-P M magnitude (see ``estimate_M``),
    but breaks this cross-pair consistency for every pair except (i, j).
    """
    df_I = load_itd()
    plot_I_omega(df_I)

    data = pd.read_csv(os.path.join(DATA_DIR, 'All_p_w_m.csv'))

    missing = sorted({f'{int(fm):02d}' for fm in data.FM.unique()} - set(abg_config))
    if missing:
        raise ValueError(f'abg_config is missing an entry for FM {missing} -- '
                         'add alpha/beta/gamma for it (see preview_alpha_beta).')

    summary = []
    all_new_M = []
    for fm in sorted(data.FM.unique()):
        df_g = real_mean_M(data, fm)
        n_p = int(df_g.P.nunique())

        # real omega grid for this ensemble, with I(omega) interpolated onto it
        omega = np.sort(df_g['W'].unique())
        W_itd, I_itd = itd_for_fm(df_I, fm)
        I_omega = np.interp(omega, W_itd, I_itd)

        cfg = abg_config[f'{int(fm):02d}']
        alpha, beta, gamma = cfg['alpha'], cfg['beta'], cfg.get('gamma', 0.0)

        # per-epoch loss trace for the mi/mj gradient descent
        _, _, df_loss = fit_mi_mj(I_itd, alpha, beta, i, j, record_loss=True, omega=W_itd)
        df_loss.to_csv(os.path.join(DATA_DIR, f'loss_val_FM_{fm}.csv'), index=False)

        # real-vs-estimated comparison uses only the matched (sparse) omega grid
        est = estimate_M(omega, I_omega, n_p, alpha, beta, gamma, i=i, j=j)
        df_cmp = pd.merge(df_g, est, on=['P', 'Wkey'], how='inner', suffixes=('', '_est'))
        df_cmp = df_cmp[['P', 'W', 'Mean_M', 'Est_M']].sort_values(['P', 'W'])
        df_cmp.to_csv(os.path.join(DATA_DIR, f'M_real_vs_est_FM_{fm}.csv'), index=False)

        # new_M is evaluated on the full ITD omega grid (0..20), not just the
        # sparse real grid used for fitting.
        est_full = estimate_M(W_itd, I_itd, n_p, alpha, beta, gamma, i=i, j=j)
        new_M = est_full[['P', 'W', 'Est_M']].copy()
        new_M['FM'] = fm
        all_new_M.append(new_M)

        plot_real_vs_synthetic(df_g, est_full, fm, alpha, beta, gamma=gamma)
        plot_real_vs_synthetic(df_g, est_full, fm, alpha, beta, full_range=True, gamma=gamma)
        plot_real_vs_synthetic_combined(df_g, est_full, fm, alpha, beta, gamma=gamma)
        plot_real_vs_synthetic_combined(df_g, est_full, fm, alpha, beta, full_range=True, gamma=gamma)

        n_matched = len(df_cmp)
        abs_diff = np.abs(df_cmp['Mean_M'] - df_cmp['Est_M'])
        mean_abs = abs_diff.mean()
        summary.append({
            'FM': fm, 'alpha': alpha, 'beta': beta, 'gamma': gamma,
            'objective': abs_diff.sum(), 'n_matched': n_matched, 'mean_abs_diff': mean_abs,
        })
        print(f'FM={fm}: alpha={alpha:.5g}, beta={beta:.5g}, '
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


def preview_alpha_beta(fm, alpha, beta, gamma=0.0, i=FIT_PAIR[0], j=FIT_PAIR[1], tag='_manual'):
    """Hand-tune (alpha, beta[, gamma]) for one FM and see the effect immediately.

    Reloads ``All_p_w_m.csv``/``rpitd/ITD.csv`` (already on disk after one
    pipeline run), builds ``estimate_M`` at
    the values you pass in, prints the per-P fit quality, and rewrites
    ``M_real_vs_synthetic*_FM_{fm}{tag}*.png`` (``tag`` defaults to ``'_manual'``
    so these never overwrite the pipeline's own ``M_real_vs_synthetic*_FM_{fm}.png``
    from ``generate_m_omega_from_I_omega``/``new_M.csv``). Call it repeatedly --
    e.g. from a notebook/REPL -- with different alpha/beta until P=1..4 line up
    well enough; nothing here writes ``new_M.csv`` or ``optimal_alpha_beta.csv``,
    so it's safe to explore without disturbing the pipeline's saved fit.

    Only ``mi``/``mj`` (P = i, j; default ``FIT_PAIR`` = 2, 3) are direct functions of
    alpha/beta -- every other P (including P=1, 4, 5) is the closed-form derived
    moment from ``estimate_M``, so alpha/beta move the other P only indirectly
    through mi/I; ``gamma`` (default 0, i.e. off) is the other knob available,
    damping just the derived moments -- see ``estimate_M``'s docstring.

    Returns ``(df_cmp, per_p)``: the matched real-vs-estimated rows (as
    ``generate_m_omega_from_I_omega`` writes to ``M_real_vs_est_FM_{fm}.csv``)
    and a per-P DataFrame of ``mean_abs_diff`` / ``rel_rms_pct`` / ``n`` so you
    can compare candidates numerically, not just by eyeballing the plots.
    """
    W_itd, I_itd = itd_for_fm(load_itd(), fm)

    data = pd.read_csv(os.path.join(DATA_DIR, 'All_p_w_m.csv'))
    df_g = real_mean_M(data, fm)
    n_p = int(df_g.P.nunique())

    omega = np.sort(df_g['W'].unique())
    I_omega = np.interp(omega, W_itd, I_itd)

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

    est_full = estimate_M(W_itd, I_itd, n_p, alpha, beta, gamma, i=i, j=j)

    plot_real_vs_synthetic(df_g, est_full, fm, alpha, beta, gamma=gamma, tag=tag)
    plot_real_vs_synthetic(df_g, est_full, fm, alpha, beta, full_range=True, gamma=gamma, tag=tag)
    plot_real_vs_synthetic_combined(df_g, est_full, fm, alpha, beta, gamma=gamma, tag=tag)
    plot_real_vs_synthetic_combined(df_g, est_full, fm, alpha, beta, full_range=True, gamma=gamma, tag=tag)

    return df_cmp, per_p


def _fit_std_model(w_real, std_real):
    """Linear  std(omega) = slope*omega + intercept, as a callable of omega.

    Only floored at a tiny positive value (so sigma stays valid); otherwise
    the line keeps growing, since the full 0..20 grid reaches ~10x the real
    omega range and capping it would defeat the extrapolation.
    """
    slope, intercept = np.polyfit(w_real, std_real, 1)
    lo = 1e-6 * float(np.max(std_real))

    def model(w):
        return np.clip(slope * np.asarray(w, dtype=float) + intercept, lo, None)

    return model


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
    plt.figure(dpi=PLOT_DPI)
    counts, bins, _ = plt.hist(x, bins=40, density=True, alpha=0.35, color='gray', label='real M')
    grid = np.linspace(bins[0], bins[-1], 400)
    for name in names:
        params = _CANDIDATE_DISTS[name].fit(x)
        plt.plot(grid, _CANDIDATE_DISTS[name].pdf(grid, *params), lw=2, label=name)
    plt.xlabel('M')
    plt.ylabel('density')
    plt.title(fr'FM={fm} $p_z$={p} $\omega$={w:.3g}  (n={len(x)})')
    plt.legend(fontsize=11)
    plt.savefig(os.path.join(IMG_DIR, f'M_distribution_example_{tag}.png'), format='png', dpi=PLOT_DPI)
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
    """Mean vs median of M per (FM,P,W).

    This only tests *symmetry* (mean==median for any symmetric distribution),
    so it cannot tell a normal from a heavier- or lighter-tailed one -- see
    ``_draw_qq_normal`` for that.
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


def plot_mean_median_and_qq(data=None):
    """Mean-vs-median and the normal Q-Q plot side by side in one figure."""
    groups = _real_groups(data)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), dpi=PLOT_DPI)
    _draw_mean_vs_median(ax1, groups, title='mean vs median')
    _draw_qq_normal(ax2, groups, title='qq-normal')
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, 'mean_median_and_qq.png'), format='png', dpi=PLOT_DPI)
    plt.close(fig)


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
        plt.figure(dpi=PLOT_DPI)
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
        plt.savefig(os.path.join(IMG_DIR, f'{out_prefix}_FM_{fm}.png'), format='png', dpi=PLOT_DPI)
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
    model (``_fit_std_model``) to the real data, then evaluates it at every
    omega in ``new_M.csv`` (0..20, includes omega beyond the real range --
    this is the extrapolation).

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
        model = _fit_std_model(std_w.index.to_numpy(dtype=float), std_w.to_numpy())

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


def generate_I_omega():
    """Per FM, fit the real per-(P, W) mean M (P in ``pl_fm[fm]``) to
        M(omega, P) = A * omega * exp(-slope * omega) * P**(-gamma) * exp(c * z2),
        z2 = min((omega / P)**2, z2_max)

    weighted by 1/SE^2 of each cell mean (omega=0 anchor rows excluded), then
    write  I(omega) = 0.05 * A * omega * exp(-slope * omega)  on omega = 0..20
    to ``rpitd/ITD.csv`` (FM, W, I) -- the input of ``generate_m_omega_from_I_omega``.
    """

    def cells(df, pl):
        g = df.groupby(["P", "W"], as_index=False).M.agg(mean="mean", std="std", n="count")
        g = g[(g.W > 0) & (g.P.isin(pl))].copy()
        g["std"] = g["std"].fillna(0.0).clip(lower=1e-8)
        g["weight"] = g["n"] / g["std"] ** 2
        return g

    def predict(w, P, A, slope, gamma, c, z2_max):
        z2 = np.minimum((w / P) ** 2, z2_max)
        return A * w * np.exp(-slope * w) * P ** (-gamma) * np.exp(c * z2)

    def fit_fm(fm, raw, pl):
        sub = raw[raw.FM == fm]
        tab = cells(sub, pl)
        z2_max = float(np.max((tab.W / tab.P) ** 2))

        def residual(th):
            A, slope, gamma, c = th
            model = predict(tab.W.to_numpy(), tab.P.to_numpy(), A, slope, gamma, c, z2_max)
            return np.sqrt(tab.weight) * (model - tab["mean"])

        fit = least_squares(
            residual,
            np.array([0.5, 0.25, 1.5, 0.1]),
            bounds=([0, 0, 0, 0], [3.0, 0.5, 3.0, 0.2]),
        )
        A, slope, gamma, c = fit.x
        model = predict(tab.W.to_numpy(), tab.P.to_numpy(), A, slope, gamma, c, z2_max)
        pull = (tab["mean"].to_numpy() - model) / (tab["std"] / np.sqrt(tab.n))
        chi2 = float(np.sum(tab.weight * (model - tab["mean"]) ** 2))
        dof = max(len(tab) - 4, 1)
        mae = float(np.mean(np.abs(model - tab["mean"])))
        return dict(FM=fm, A=A, slope=slope, gamma=gamma, c=c, z2_max=z2_max,
                    chi2_dof=chi2 / dof, mae=mae, pull_min=float(pull.min()),
                    pull_max=float(pull.max()), n_cells=len(tab))

    df_real = pd.read_csv(os.path.join(DATA_DIR, 'All_p_w_m.csv'))
    I_parts = []
    omega = np.round(np.arange(0.0, 20.0 + 1e-9, 0.1), 10)
    pl_fm = {'9': [3], '12': [3], '15': [3]}   # P values each FM's fit uses

    for fm in sorted(df_real.FM.unique()):
        result = fit_fm(fm, df_real, pl_fm[str(fm)])

        I = result["A"] * omega * np.exp(-result["slope"] * omega) * 0.05
        I_parts.append(pd.DataFrame({'FM': int(fm), 'W': omega, 'I': I}))

        print(
            f"FM {fm}: I_max={max(I):.3f}  A={result['A']:.3f} slope={result['slope']:.3f} "
            f"gamma={result['gamma']:.3f} c={result['c']:.4f} "
            f"chi2/dof={result['chi2_dof']:.0f} MAE={result['mae']:.4f} "
            f"pull [{result['pull_min']:.0f}, {result['pull_max']:.0f}]"
        )

    pd.concat(I_parts, ignore_index=True).to_csv(os.path.join(DATA_DIR, 'rpitd', 'ITD.csv'), index=False)


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Default (no arguments): run the full synthetic-data pipeline '
                    '(alpha/beta per FM from abg_config). '
                    'Pass --fm together with --alpha/--beta to instead preview a '
                    'hand-picked fit for that one FM via preview_alpha_beta, without '
                    'touching new_M.csv or optimal_alpha_beta.csv.')
    parser.add_argument('--fm', type=int, default=None,
                        help='Ensemble id (e.g. 9, 12, 15). Requires --alpha and --beta.')
    parser.add_argument('--alpha', type=float, default=None, help='alpha to preview for --fm.')
    parser.add_argument('--beta', type=float, default=None, help='beta to preview for --fm.')
    parser.add_argument('--gamma', type=float, default=0.0,
                        help='Optional high-P damping term (see estimate_M). Default 0 (off).')
    parser.add_argument('--i', type=int, default=FIT_PAIR[0],
                        help='Lower momentum of the directly-fit pair (>= 2; P=1 is never used).')
    parser.add_argument('--j', type=int, default=FIT_PAIR[1], help='Higher momentum of the directly-fit pair.')
    parser.add_argument('--tag', default='_manual',
                        help="Suffix for the preview plot filenames, so they don't overwrite "
                             "the pipeline's own M_real_vs_synthetic*_FM_{fm}.png.")
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    if args.fm is not None or args.alpha is not None or args.beta is not None:
        if args.fm is None or args.alpha is None or args.beta is None:
            raise SystemExit('--fm, --alpha and --beta must all be given together.')
        preview_alpha_beta(args.fm, args.alpha, args.beta, gamma=args.gamma,
                           i=args.i, j=args.j, tag=args.tag)
    else:
        # always rebuild (cheap) so a stale All_p_w_m.csv never outlives a
        # change to N_OMEGA_PER_FM
        data = create_combined_data()

        generate_I_omega()
        generate_m_omega_from_I_omega()

        # real-data diagnostics (CSVs/plots only)
        check_M_distribution()
        check_linearity(data)
        plot_mean_vs_w(data)
        plot_std_vs_w(data)
        plot_mean_median_and_qq(data)

        generate_m_omega_from_distribution()
