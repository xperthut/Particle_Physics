"""Continuum limit (a -> 0) taken on M(omega, P, a), then I(omega) with a confidence band.

Unlike continuum_extrapolation.py (pointwise fit of the final, smoothed I), the
cutoff dependence is removed where it arises -- on the matrix element -- and
I(omega) is built afterwards from the continuum M with the same momentum-pair
relation test_error.py uses.

Input, per lattice spacing FM in {9, 12, 15} (a = 0.0888, 0.1207, 0.151 fm):
    --source anchored (default)
        Data/2026/Real_data_projected_{fm}_no_exp_anchored_v2.csv     (P, Exp, W, M_<model>)
        Data/2026/Real_data_projected_{fm}_no_exp_anchored_draws_v2.csv (P, Draw, W, M_GP)
        from `reconstruct_real_data.py --anchor_real --n_draws K --models GP`:
        each Exp follows its own real data over the measured range, and the GP
        latent draws carry the model uncertainty of the extrapolated tail.  Each
        bootstrap sample adds one draw's shift (draw k - draw 0), chosen
        independently per FM since every FM has its own GP.
    --source roll
        Data/2026/Real_data_projected_{fm}_no_exp_v2.csv  (plain seed-only roll)
the ML-projected M on the 0.1 omega grid, one curve per gauge configuration Exp.
The model only interpolates/extends in omega; the a -> 0 step is a parametric fit.

Global fit, one per (P, form), over every omega > 0 and all three spacings at once:

    M(omega, a) = M0(omega) + sum_k theta_k * g_k(omega, a)

M0(omega) is free at every omega (the continuum curve); the cutoff term is shared
across omega and vanishes at omega = 0 (M(0) = 0 for every a).  Three forms:

    'a'    g = a      * (x, x^2)       O(a): unimproved bilocal operator
    'a2'   g = a^2    * (x, x^2)       O(a^2): fully O(a)-improved
    'a_z'  g = (a/z)  * (x^2, x^3)     O(a/z) = O(1/z_lattice), z = omega / (2 pi P / L)
with x = omega / omega_max.  On these ensembles L*a is nearly constant (2.84,
2.90, 2.42 fm), so 'a' and 'a_z' are close to degenerate -- a 4-parameter
a + a/z form is not resolvable and is not offered.

Why the fit is unweighted: the projected replicas collapse onto one curve away
from the lattice data (their Exp spread falls to ~1e-5 and to 0 by omega ~ 20),
while the spacings differ by ~1e-2.  Error-of-the-mean weights would give
chi^2/dof ~ 1e9 and a meaningless one-hot model choice.  The fit is therefore
ordinary least squares and its uncertainty is taken from its own residuals,
i.e. from how well the cutoff form actually describes the three spacings.

Uncertainty (one combined sample per bootstrap draw b = 1..n_boot):
  * configurations -- resample Exp with replacement independently per FM
    (separate ensembles), average, refit;
  * fit            -- add a draw from N(0, s^2 (X^T X)^-1) on M0, s^2 = RSS/dof;
  * I(omega) is rebuilt from each sample; 'stat' = bootstrap 68% half-width;
  * cutoff form    -- 'sys' = weighted spread of the central I across forms.
  Total sigma = stat (+) sys in quadrature; bands at 68% (1 sigma) and 95% (1.96 sigma).
Form weights (--weights): 'equal' (default) or 'aic' (n ln(RSS/n) + 2k).  The
residuals are strongly correlated in omega (the curves are smooth), so AIC's
independent-point assumption overstates the evidence and tends to pick one form
outright; 'equal' keeps the form ambiguity in the band.  Both are printed.

I(omega) per sample: every pair 2 <= p1 < p2 gives
    I = (p2^2 M(p2) - p1^2 M(p1)) / (p2^2 - p1^2),
non-negative pair estimates are averaged, then Savitzky-Golay smoothed, I(0) = 0 --
identical to test_error.py so the result is comparable with the per-FM curves.

Writes
    Data/2026/I_omega_continuum_M_{model}.csv
        omega, I, I_lo68, I_hi68, I_lo95, I_hi95, sigma_stat, sigma_sys,
        I_<form> (central, per form), I_FM<fm> (per-spacing, same recipe)
    Data/2026/M_continuum_fit_{model}.csv     per-(P, form) RMS residual, AIC, theta
    Data/2026/Images/I_omega_continuum_M_{model}.{png,pdf}

The model is the overall #1 (FM == 'all') in Data/2026/train_test_error_no_exp.csv;
latent draws exist only for GP, otherwise the band is configurations + fit only.

Run from Helicity/Code:  python continuum_M_extrapolation.py [--n_boot 500]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")            # script context: render to files, no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

from test_error import MIN_P, SMOOTH_WINDOW, best_model, get_smooth_curve

FM_TO_A = {9: 0.0888, 12: 0.1207, 15: 0.151}     # fm, from the rpitd file headers
W_DECIMALS = 1
FORMS = ('a', 'a2', 'a_z')
FORM_LABEL = {'a': r'$O(a)$', 'a2': r'$O(a^2)$', 'a_z': r'$O(a/z)$'}
COLORS = ['#2a78d6', '#eb6834', '#1baf7a']
STYLES = ['-', '--', '-.']


# --------------------------------------------------------------------------- data

def load_replicas(data_dir, model, suffix):
    """{fm: array (n_P, n_Exp, n_W)} of projected M, plus the shared P and W grids."""
    out, grid_P, grid_W = {}, None, None
    for fm in FM_TO_A:
        path = os.path.join(data_dir, f'Real_data_projected_{fm}_no_exp{suffix}_v2.csv')
        d = pd.read_csv(path, usecols=['P', 'Exp', 'W', f'M_{model}'])
        d['W'] = np.round(d['W'], W_DECIMALS)
        P, W = np.sort(d.P.unique()), np.sort(d.W.unique())
        if grid_P is None:
            grid_P, grid_W = P, W
        elif not (np.array_equal(P, grid_P) and np.allclose(W, grid_W)):
            raise ValueError(f'FM={fm}: P/W grid differs from the other spacings')
        cube = (d.set_index(['P', 'Exp', 'W'])[f'M_{model}']
                .unstack('W').reindex(columns=W))
        exps = np.sort(d.Exp.unique())
        out[fm] = np.stack([cube.loc[p].reindex(exps).to_numpy() for p in P])
        print(f'FM={fm}: {len(exps)} Exp replicas, P={list(P)}, {len(W)} omega points')
    return out, grid_P, grid_W


def load_draw_shifts(data_dir, model, suffix, P, W):
    """{fm: (n_draws, n_P, n_W)} GP latent-draw shifts (draw k - draw 0), or None if absent."""
    out = {}
    for fm in FM_TO_A:
        path = os.path.join(data_dir, f'Real_data_projected_{fm}_no_exp{suffix}_draws_v2.csv')
        if model != 'GP' or not os.path.exists(path):
            return None
        d = pd.read_csv(path)
        d['W'] = np.round(d['W'], W_DECIMALS)
        cube = (d.set_index(['Draw', 'P', 'W'])[f'M_{model}'].unstack('W')
                .reindex(columns=W))
        draws = np.sort(d.Draw.unique())
        arr = np.stack([cube.loc[k].reindex(P).to_numpy() for k in draws])
        out[fm] = arr[1:] - arr[0]
        print(f'FM={fm}: {len(draws) - 1} GP latent draws')
    return out


def lattice_units(data_dir):
    """{(fm, P): omega per lattice z step (= 2 pi P / L)} and the real-data omega reach per FM."""
    r = pd.read_csv(os.path.join(data_dir, 'All_p_w_m.csv'), usecols=['FM', 'P', 'Z', 'W'])
    r = r[r.Z > 0]
    unit = (r.W / r.Z).groupby([r.FM, r.P]).median().to_dict()
    reach = r.groupby('FM').W.max().to_dict()
    return unit, reach


# --------------------------------------------------------------------------- fit

def basis(form, a, unit, omega, x):
    """(n_FM*n_W, 2) cutoff basis, rows ordered FM-major to match the stacked data."""
    cols = []
    for af, uf in zip(a, unit):
        if form == 'a':
            s = af * np.ones_like(x)
        elif form == 'a2':
            s = af ** 2 * np.ones_like(x)
        else:                                # a/z_phys = 1 / z_lattice = unit / omega
            s = uf / omega
        # 'a_z' carries an extra x so the term still vanishes at omega -> 0
        g = np.column_stack([x, x ** 2]) * (x[:, None] if form == 'a_z' else 1.0)
        cols.append(s[:, None] * g)
    return np.vstack(cols)


def design(form, a, unit, omega, x):
    """Design matrix [M0 per omega | shared theta] and its least-squares operator."""
    n_fm, n_w = len(a), len(omega)
    X = np.hstack([np.tile(np.eye(n_w), (n_fm, 1)), basis(form, a, unit, omega, x)])
    return X, np.linalg.pinv(X)


# --------------------------------------------------------------------------- I(omega)

def pair_I(M, P):
    """I(omega) from M (..., n_P, n_W): mean of non-negative pair estimates, P >= MIN_P."""
    ests = []
    for i, p1 in enumerate(P):
        for j, p2 in enumerate(P):
            if MIN_P <= p1 < p2:
                ests.append((p2 ** 2 * M[..., j, :] - p1 ** 2 * M[..., i, :]) / (p2 ** 2 - p1 ** 2))
    E = np.stack(ests)
    E = np.where(E >= 0, E, np.nan)
    with np.errstate(invalid='ignore'):
        I = np.nanmean(E, axis=0)
    return np.nan_to_num(I, nan=0.0)


def smooth_I(I, omega, window):
    sm = np.apply_along_axis(get_smooth_curve, -1, I, window)
    sm[..., omega == 0.0] = 0.0
    return np.clip(sm, 0.0, None)


# --------------------------------------------------------------------------- main

def run(data_dir, model, n_boot, seed, window, weighting, source):
    suffix = '_anchored' if source == 'anchored' else ''
    reps, P, W = load_replicas(data_dir, model, suffix)
    shifts = load_draw_shifts(data_dir, model, suffix, P, W)
    unit_map, reach = lattice_units(data_dir)
    fms = list(FM_TO_A)
    a = np.array([FM_TO_A[f] for f in fms])

    fit = W > 0                                   # M(0) = 0 exactly at every a
    omega = W[fit]
    x = omega / omega.max()
    n_w = len(omega)

    mean = np.stack([reps[f].mean(axis=1) for f in fms])       # (n_FM, n_P, n_W)

    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, len(fms), len(P), len(W)))        # bootstrap Exp-means
    for f_i, f in enumerate(fms):
        n_exp = reps[f].shape[1]
        for b in range(n_boot):
            boot[b, f_i] = reps[f][:, rng.integers(0, n_exp, n_exp), :].mean(axis=1)
        if shifts is not None:              # + one GP latent draw (independent model per FM)
            boot[:, f_i] += shifts[f][rng.integers(0, len(shifts[f]), n_boot)]

    M0_c, M0_b, fit_rows = {}, {}, []
    for form in FORMS:
        M0_c[form] = np.zeros((len(P), len(W)))
        M0_b[form] = np.zeros((n_boot, len(P), len(W)))
        for k, p in enumerate(P):
            unit = np.array([unit_map[(f, p)] for f in fms])
            X, S = design(form, a, unit, omega, x)
            y = mean[:, k, fit].ravel()
            coef = S @ y
            rss = float(np.sum((y - X @ coef) ** 2))
            n, n_par = len(y), X.shape[1]
            s2 = rss / (n - n_par)
            M0_c[form][k, fit] = coef[:n_w]

            # configuration bootstrap refit + fit-residual draw on M0
            yb = boot[:, :, k, :][:, :, fit].reshape(n_boot, -1)
            cov = s2 * (S @ S.T)[:n_w, :n_w]                  # = s^2 (X^T X)^-1 block
            evals, evecs = np.linalg.eigh(cov)
            L = evecs * np.sqrt(np.clip(evals, 0.0, None))
            M0_b[form][:, k, fit] = (yb @ S.T)[:, :n_w] + rng.standard_normal((n_boot, n_w)) @ L.T

            fit_rows.append({'P': p, 'form': form, 'n': n, 'n_par': n_par,
                             'RMS_resid': np.sqrt(rss / n),
                             'AIC': n * np.log(rss / n) + 2 * n_par,
                             'theta': coef[n_w:].round(5).tolist()})

    fits = pd.DataFrame(fit_rows)
    # forms used for I(omega): only P >= MIN_P enter the pair relation
    aic = fits[fits.P >= MIN_P].groupby('form').AIC.sum().reindex(FORMS)
    w_aic = np.exp(-(aic - aic.min()) / 2)
    w_aic = w_aic / w_aic.sum()
    w_form = w_aic if weighting == 'aic' else pd.Series(1.0 / len(FORMS), index=FORMS)
    fits['weight'] = fits.form.map(w_form)
    print('\nRMS fit residual per P and form (spacing mismatch the form cannot absorb):')
    print(fits.pivot(index='P', columns='form', values='RMS_resid').to_string(float_format='%.2e'))
    print(f'\nsum AIC over P>={MIN_P}:', aic.round(1).to_dict())
    print('AIC weights:', w_aic.round(4).to_dict(), f'| using {weighting} weights:',
          w_form.round(3).to_dict())

    I_c = {f: smooth_I(pair_I(M0_c[f], P), W, window) for f in FORMS}
    I_b = {f: smooth_I(pair_I(M0_b[f], P), W, window) for f in FORMS}

    wv = w_form.to_numpy()
    I_avg = sum(wv[i] * I_c[f] for i, f in enumerate(FORMS))
    I_avg_b = sum(wv[i] * I_b[f] for i, f in enumerate(FORMS))
    lo, hi = np.percentile(I_avg_b, [16, 84], axis=0)
    s_stat = (hi - lo) / 2
    s_sys = np.sqrt(sum(wv[i] * (I_c[f] - I_avg) ** 2 for i, f in enumerate(FORMS)))
    s_tot = np.hypot(s_stat, s_sys)

    # per-spacing curves with the identical recipe, for comparison
    I_fm = {f: smooth_I(pair_I(mean[i], P), W, window) for i, f in enumerate(fms)}

    clip = lambda v: np.clip(v, 0.0, None)
    out = pd.DataFrame({'omega': W, 'I': I_avg,
                        'I_lo68': clip(I_avg - s_tot), 'I_hi68': I_avg + s_tot,
                        'I_lo95': clip(I_avg - 1.96 * s_tot), 'I_hi95': I_avg + 1.96 * s_tot,
                        'sigma_stat': s_stat, 'sigma_sys': s_sys,
                        **{f'I_{f}': I_c[f] for f in FORMS},
                        **{f'I_FM{f}': I_fm[f] for f in fms}})
    out_csv = os.path.join(data_dir, f'I_omega_continuum_M_{model}{suffix}.csv')
    out.to_csv(out_csv, index=False)
    fit_csv = os.path.join(data_dir, f'M_continuum_fit_{model}{suffix}.csv')
    fits.to_csv(fit_csv, index=False)
    print(f'\ncontinuum I(omega) -> {out_csv}\nfit summary -> {fit_csv}')
    i_pk = int(np.argmax(I_avg))
    print(f'peak I = {I_avg[i_pk]:.4f} +- {s_tot[i_pk]:.4f} (stat {s_stat[i_pk]:.4f}, '
          f'sys {s_sys[i_pk]:.4f}) at omega = {W[i_pk]:.1f}')

    band = ('per-configuration real data + GP latent draws' if shifts is not None
            else 'configuration bootstrap') + r' + fit residual $\oplus$ cutoff-form spread'
    plot(out, fms, a, w_form, weighting, max(reach.values()), model, band,
         os.path.join(data_dir, 'Images', f'I_omega_continuum_M_{model}{suffix}'))


def plot(out, fms, a, w_form, weighting, w_real, model, band, out_base):
    w = out.omega.to_numpy()
    fig, ax = plt.subplots(figsize=(9, 6), dpi=330)
    for k, (fm, av) in enumerate(zip(fms, a)):
        ax.plot(w, out[f'I_FM{fm}'], STYLES[k % len(STYLES)], lw=1.2, alpha=0.8,
                color=COLORS[k % len(COLORS)], label=f'$a={av:.3f}$ fm (FM={fm})')
    ax.fill_between(w, out.I_lo95, out.I_hi95, color='0.2', alpha=0.12, lw=0, label='95% CI')
    ax.fill_between(w, out.I_lo68, out.I_hi68, color='0.2', alpha=0.28, lw=0, label='68% CI')
    wtxt = ', '.join(f'{FORM_LABEL[f]} {w_form[f]:.2f}' for f in FORMS)
    ax.plot(w, out.I, '-', lw=2.4, color='0.05',
            label=f'continuum $a\\to0$ ({weighting} weights: {wtxt})')
    ax.axvline(w_real, ls=':', lw=1, color='0.5')
    ax.annotate('lattice data reach', (w_real, 0), xytext=(4, 4), textcoords='offset points',
                fontsize=8, color='0.4')

    ax.set_xlim(w.min(), w.max())
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'Ioffe time $\omega$')
    ax.set_ylabel(r'$\Delta\mathcal{I}_g(\omega)$')
    ax.set_title(f'Continuum limit on $M(\\omega,P,a)$, then $I(\\omega)$  [{model}, no Exp feature]\n'
                 f'band: {band}', fontsize=10)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(alpha=0.25, lw=0.5)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out_base}.{ext}', bbox_inches='tight')
        print(f'plot -> {out_base}.{ext}')
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='../Data/2026')
    parser.add_argument("--n_boot", type=int, default=500, help="bootstrap samples over Exp")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--weights", choices=['equal', 'aic'], default='equal',
                        help="how the cutoff forms are combined")
    parser.add_argument("--smooth_window", type=int, default=SMOOTH_WINDOW,
                        help="Savitzky-Golay window for I(omega), as in test_error.py")
    parser.add_argument("--source", choices=['anchored', 'roll'], default='anchored',
                        help="anchored: reconstruct_real_data.py --anchor_real output (+ GP draws "
                             "if present); roll: the plain seed-only roll")
    args = parser.parse_args()
    run(args.data_dir, best_model(args.data_dir), args.n_boot, args.seed, args.smooth_window, args.weights,
        args.source)
