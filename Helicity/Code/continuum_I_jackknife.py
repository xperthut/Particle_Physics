"""Continuum limit (a -> 0) of I(omega) with jackknife errors over the configurations.

The Exp rows in the projected files are jackknife samples (jackknife.py).  Per lattice
spacing FM in {9, 12, 15} (a = 0.0888, 0.1207, 0.151 fm):

  1. build I(omega) from the Exp-mean projected M with the momentum-pair
     relation of test_error.py,
         I = (p2^2 M(p2) - p1^2 M(p1)) / (p2^2 - p1^2),  pairs in test_error.I_PAIRS,
     averaged over the pairs and Savitzky-Golay smoothed (same window/order),
     and the same on every Exp row (each row is a jackknife sample), giving
     the per-FM jackknife error sqrt(N-1) * std (jackknife.py);
  2. at every omega fit the three spacings with
         I(omega, a) = I0(omega) + c(omega) * a^2          (main result)
         I(omega, a) = I0(omega) + c(omega) * a            (systematic check)
     a^2 is the leading cutoff term for the O(a)-improved clover valence action
     on the HISQ sea; with three spacings a two-parameter fit is all the data
     can constrain.

The ensembles are independent, so each FM is jackknifed on its own with the
other two at their mean; the fit is linear in the I_FM, so
    sigma_stat^2 = sum_FM (dI0/dI_FM)^2 sigma_JK,FM^2
is exact.  Systematic = |I0_a2 - I0_a|.  Bands: dark = jackknife statistical,
light = statistical (+) systematic in quadrature.

Model systematic, kept separate (--sys_models, e.g. --model RF --sys_models NN):
each alternative is extrapolated the same way (a^2 fit) and
sigma_model = |I0_model - I0_alt| is written to its own CSV/figure; the
cutoff output above is unchanged by it.  Pair estimates are not
clipped at 0 (that would bias the jackknife); only the plotted curve is floored.

The model is --model, or by default the overall #1 in Data/2026/test_error_ranking_no_exp.csv.

Input:  Data/2026/Real_data_projected_{fm}_no_exp{suffix}_v2.csv  (P, Exp, W, M_<model>)
        --source roll (default: the plain roll) or anchored (--anchor_real output)
Writes: Data/2026/I_omega_continuum_jk_{model}{suffix}.csv
            omega, I, sigma_stat, sigma_sys, sigma_tot, I_lin_a, c2, I_FM<fm>, sigma_FM<fm>
        Data/2026/Images/I_omega_continuum_jk_{model}{suffix}.{png,pdf}
        with --sys_models <alt>:
        Data/2026/I_omega_continuum_jk_{model}_vs_{alt}{suffix}.csv
            omega, I_<model>, sigma_stat_<model>, I_<alt>, sigma_stat_<alt>, sigma_model
        Data/2026/Images/I_omega_continuum_jk_{model}_vs_{alt}{suffix}.{png,pdf}

Run from Helicity/Code:  python continuum_I_jackknife.py [--source roll] [--model RF]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")            # script context: render to files, no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

from jackknife import jk_error
from test_error import I_PAIRS, MODEL_NAMES, SMOOTH_WINDOW, best_model, get_smooth_curve

FM_TO_A = {9: 0.0888, 12: 0.1207, 15: 0.151}     # fm, from the rpitd file headers
W_DECIMALS = 1
COLORS = ['#2a78d6', '#eb6834', '#1baf7a']
STYLES = ['-', '--', '-.']


def load_samples(path, model):
    """(P, W, M[n_P, n_Exp, n_W]) from one projected file."""
    d = pd.read_csv(path, usecols=['P', 'Exp', 'W', f'M_{model}'])
    d['W'] = np.round(d['W'], W_DECIMALS)
    P, W, E = np.sort(d.P.unique()), np.sort(d.W.unique()), np.sort(d.Exp.unique())
    cube = d.set_index(['P', 'Exp', 'W'])[f'M_{model}'].unstack('W').reindex(columns=W)
    return P, W, np.stack([cube.loc[p].reindex(E).to_numpy() for p in P])


def pair_I(M, P, W, window):
    """I(omega) per sample from M (n_P, n_samples, n_W): mean over pairs, smoothed, I(0) = 0."""
    idx = {p: k for k, p in enumerate(P)}
    ests = [(p2 ** 2 * M[idx[p2]] - p1 ** 2 * M[idx[p1]]) / (p2 ** 2 - p1 ** 2)
            for p1, p2 in I_PAIRS if p1 in idx and p2 in idx]
    if not ests:
        raise ValueError(f'none of the I_PAIRS {I_PAIRS} present (P = {list(P)})')
    I = np.mean(ests, axis=0)                                   # (n_samples, n_W)
    I = np.apply_along_axis(get_smooth_curve, 1, I, window)
    I[:, W == 0.0] = 0.0
    return I


def extrapolate_op(a, power):
    """(2, n_FM) least-squares operator: [I0, c] = S @ I_FM for I = I0 + c * a**power."""
    return np.linalg.pinv(np.column_stack([np.ones_like(a), a ** power]))


def per_fm_I(data_dir, model, suffix, fms, window):
    """(W, I_FM[n_FM, n_W] on the Exp mean, jackknife sigma_FM[n_FM, n_W]) for one model."""
    I_fm, s_fm, W = [], [], None
    for fm in fms:
        path = os.path.join(data_dir, f'Real_data_projected_{fm}_no_exp{suffix}_v2.csv')
        P, W_f, M = load_samples(path, model)
        if W is not None and not np.allclose(W, W_f):
            raise ValueError(f'FM={fm}: omega grid differs from the other spacings')
        W = W_f
        I_fm.append(pair_I(np.nanmean(M, axis=1, keepdims=True), P, W, window)[0])
        s_fm.append(jk_error(pair_I(M, P, W, window), axis=0))
        print(f'[{model}] FM={fm}: {M.shape[1]} configurations (jackknife), P={list(P)}')

        cum_sum = True
        if cum_sum and model != 'RF':
            #print(model)
            s_fm[-1] = np.sqrt(np.cumsum(s_fm[-1] ** 2))

    return W, np.stack(I_fm), np.stack(s_fm)                    # (n_FM, n_W)


def continuum(data_dir, model, suffix, fms, a, window):
    """Per-model continuum frame: a^2 fit (central), a fit, jackknife stat, cutoff sys."""
    S2, S1 = extrapolate_op(a, 2), extrapolate_op(a, 1)
    W, I_fm, s_fm = per_fm_I(data_dir, model, suffix, fms, window)
    I0_a2, c2 = S2 @ I_fm
    I0_a1 = (S1 @ I_fm)[0]
    s_stat = np.sqrt(((S2[0][:, None] * s_fm) ** 2).sum(axis=0))
    s_sys = np.abs(I0_a2 - I0_a1)                   # cutoff systematic: a^2 vs a
    return pd.DataFrame({'omega': W, 'I': I0_a2, 'sigma_stat': s_stat,
                         'sigma_sys': s_sys, 'sigma_tot': np.hypot(s_stat, s_sys),
                         'I_lin_a': I0_a1, 'c2': c2,
                         **{f'I_FM{f}': I_fm[k] for k, f in enumerate(fms)},
                         **{f'sigma_FM{f}': s_fm[k] for k, f in enumerate(fms)}})


def run(data_dir, model, source, window, sys_models=()):
    """Cutoff systematic (a^2 vs a) of ``model`` -> I_omega_continuum_jk_<model>.

    Each ``sys_models`` entry gives a separate model-systematic output
    I_omega_continuum_jk_<model>_vs_<alt>: sigma_model = |I0_model - I0_alt|,
    both a^2 fits.  The two systematics are kept in separate files/figures.
    """
    suffix = '_anchored' if source == 'anchored' else ''
    fms = list(FM_TO_A)
    a = np.array([FM_TO_A[f] for f in fms])

    out = continuum(data_dir, model, suffix, fms, a, window)
    out_csv = os.path.join(data_dir, f'I_omega_continuum_jk_{model}{suffix}.csv')
    out.to_csv(out_csv, index=False)
    print(f'continuum I(omega) -> {out_csv}')
    i = int(out.I.idxmax())
    print(f'peak I = {out.I[i]:.4f} +- {out.sigma_stat[i]:.4f} (stat, jackknife) '
          f'+- {out.sigma_sys[i]:.4f} (sys) at omega = {out.omega[i]:.1f}')
    plot(out, fms, a, model, source,
         os.path.join(data_dir, 'Images', f'I_omega_continuum_jk_{model}{suffix}'))

    for alt in sys_models:
        if alt == model:
            continue
        alt_out = continuum(data_dir, alt, suffix, fms, a, window)
        if not np.allclose(out.omega, alt_out.omega):
            raise ValueError(f'{alt}: omega grid differs from {model}')
        ms = pd.DataFrame({'omega': out.omega,
                           f'I_{model}': out.I, f'sigma_stat_{model}': out.sigma_stat,
                           f'I_{alt}': alt_out.I, f'sigma_stat_{alt}': alt_out.sigma_stat,
                           'sigma_model': (out.I - alt_out.I).abs()})
        base = f'I_omega_continuum_jk_{model}_vs_{alt}{suffix}'
        ms_csv = os.path.join(data_dir, f'{base}.csv')
        ms.to_csv(ms_csv, index=False)
        print(f'model systematic {model} vs {alt} -> {ms_csv}')
        W = ms.omega.to_numpy()
        for lo, hi in [(0, 5), (5, 10), (10, 15), (15, 20.01)]:
            m = (W >= lo) & (W < hi)
            print(f'  omega [{lo:>2}, {hi:>2.0f}): median stat_{model}='
                  f'{ms[f"sigma_stat_{model}"][m].median():.4f}  '
                  f'model |{model}-{alt}|={ms.sigma_model[m].median():.4f}')
        plot_model_sys(ms, model, alt, source, os.path.join(data_dir, 'Images', base))


def plot(out, fms, a, model, source, out_base):
    w = out.omega.to_numpy()
    floor = lambda v: np.clip(v, 0.0, None)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=330)
    for k, (fm, av) in enumerate(zip(fms, a)):
        ax.plot(w, floor(out[f'I_FM{fm}']), STYLES[k % len(STYLES)], lw=1.2, alpha=0.85,
                color=COLORS[k % len(COLORS)], label=f'$a={av:.3f}$ fm (FM={fm})')
    ax.fill_between(w, floor(out.I - out.sigma_tot), floor(out.I + out.sigma_tot),
                    color='0.2', alpha=0.13, lw=0,
                    label='Stat + systematic error')
    ax.fill_between(w, floor(out.I - out.sigma_stat), floor(out.I + out.sigma_stat),
                    color='0.2', alpha=0.32, lw=0, label='Statistical error')
    ax.plot(w, floor(out.I), '-', lw=2.4, color='0.05',
            label=r'continuum $a\to0$: $I_0 + c\,a^{2}$')

    ax.set_xlim(w.min(), w.max())
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'Ioffe time $\omega$')
    ax.set_ylabel(r'$\Delta\mathcal{I}_g(\omega)$')
    ax.set_title(f'Continuum extrapolation of $I(\\omega)$, jackknife over configurations  '
                 f'[{model}, {source}]', fontsize=10)
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


def plot_model_sys(ms, model, alt, source, out_base):
    """Model systematic only: both models' a^2 continuum curves (each with its
    jackknife band) and the |I0_model - I0_alt| band around ``model``."""
    w = ms.omega.to_numpy()
    floor = lambda v: np.clip(v, 0.0, None)
    I, Ia = ms[f'I_{model}'], ms[f'I_{alt}']
    sm, sa = ms[f'sigma_stat_{model}'], ms[f'sigma_stat_{alt}']
    c_m, c_a = '0.05', '#8a3ffc'
    fig, ax = plt.subplots(figsize=(9, 6), dpi=330)
    ax.fill_between(w, floor(I - ms.sigma_model), floor(I + ms.sigma_model),
                    facecolor='none', edgecolor='#eb6834', hatch='///', lw=0,
                    label=f'model systematic $|I_0^{{{model}}} - I_0^{{{alt}}}|$')
    ax.fill_between(w, floor(I - sm), floor(I + sm), color=c_m, alpha=0.25, lw=0,
                    label=f'{model} statistical error')
    ax.fill_between(w, floor(Ia - sa), floor(Ia + sa), color=c_a, alpha=0.18, lw=0,
                    label=f'{alt} statistical error')
    ax.plot(w, floor(I), '-', lw=2.4, color=c_m, label=f'{model} continuum (central)')
    ax.plot(w, floor(Ia), '--', lw=1.8, color=c_a, label=f'{alt} continuum')

    ax.set_xlim(w.min(), w.max())
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'Ioffe time $\omega$')
    ax.set_ylabel(r'$\Delta\mathcal{I}_g(\omega)$')
    ax.set_title(f'Model systematic of the continuum $I(\\omega)$ ($a^2$ fit): '
                 f'{model} vs {alt}  [{source}]', fontsize=10)
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
    parser.add_argument("--source", choices=['roll', 'anchored'], default='roll',
                        help="roll: plain projection; anchored: reconstruct_real_data.py --anchor_real")
    parser.add_argument("--smooth_window", type=int, default=SMOOTH_WINDOW,
                        help="Savitzky-Golay window for I(omega), as in test_error.py")
    parser.add_argument("--model", choices=MODEL_NAMES, default=None,
                        help="GB|RF|XGB|NN|GP (default: overall #1 in test_error_ranking_no_exp.csv)")
    parser.add_argument("--sys_models", nargs='+', choices=MODEL_NAMES, default=[],
                        help="Also write a separate model-systematic output per model: "
                             "sigma_model = |I0_model - I0_alt| (cutoff output unchanged)")
    args = parser.parse_args()
    run(args.data_dir, best_model(args.data_dir, model=args.model), args.source,
        args.smooth_window, sys_models=args.sys_models)
