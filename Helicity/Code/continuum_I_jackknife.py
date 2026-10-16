"""Continuum limit (a -> 0) of I(omega), bootstrap sample by bootstrap sample.

The Exp index in the projected files is a bootstrap sample.  For each sample:

  1. at every lattice spacing FM in {9, 12, 15} (a = 0.0888, 0.1207, 0.151 fm)
     build I(omega) from that sample's projected M with the momentum-pair
     relation of test_error.py,
         I = (p2^2 M(p2) - p1^2 M(p1)) / (p2^2 - p1^2),  every pair 2 <= p1 < p2,
     averaged over the pairs and Savitzky-Golay smoothed (same window/order);
  2. at every omega fit the three spacings with
         I(omega, a) = I0(omega) + c(omega) * a^2          (main result)
         I(omega, a) = I0(omega) + c(omega) * a            (systematic check)
     a^2 is the leading cutoff term for the O(a)-improved clover valence action
     on the HISQ sea; with three spacings a two-parameter fit is all the data
     can constrain.

Then over the samples: central I0 = mean, statistical error = std (bootstrap
samples, so no sqrt(N) rescaling).  Systematic = |I0_a2 - I0_a| on the means.
Bands: dark = 68% statistical, light = statistical (+) systematic in quadrature.

The ensembles are independent, so sample b of one FM is paired with sample b of
the others; the counts differ (1026 / 1013 / 900), so the first N_min are used.
Pair estimates are NOT clipped at 0 per sample (that would bias the mean up);
only the plotted curve is floored at 0.

Input:  Data/2026/Real_data_projected_{fm}_no_exp{suffix}_v2.csv  (P, Exp, W, M_<model>)
        --source roll (default: the plain roll) or anchored (--anchor_real output)
Writes: Data/2026/I_omega_continuum_boot_{model}{suffix}.csv
            omega, I, sigma_stat, sigma_sys, sigma_tot, I_lin_a, c2, I_FM<fm>
        Data/2026/Images/I_omega_continuum_boot_{model}{suffix}.{png,pdf}

The model is the overall #1 (FM == 'all') in Data/2026/train_test_error_no_exp.csv.

Run from Helicity/Code:  python continuum_I_bootstrap.py [--source roll]
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
    """I(omega) per sample from M (n_P, n_Exp, n_W): mean over pairs, smoothed, I(0) = 0."""
    ests = [(p2 ** 2 * M[j] - p1 ** 2 * M[i]) / (p2 ** 2 - p1 ** 2)
            for i, p1 in enumerate(P) for j, p2 in enumerate(P) if MIN_P <= p1 < p2]
    I = np.mean(ests, axis=0)                                   # (n_Exp, n_W)
    I = np.apply_along_axis(get_smooth_curve, 1, I, window)
    I[:, W == 0.0] = 0.0
    return I


def extrapolate(a, I, power):
    """I: (n_FM, n_samples, n_W) -> (I0, c), each (n_samples, n_W), by least squares in a**power."""
    X = np.column_stack([np.ones_like(a), a ** power])
    S = np.linalg.pinv(X)                                       # (2, n_FM)
    coef = np.einsum('kf,fsw->ksw', S, I)
    return coef[0], coef[1]


def run(data_dir, model, source, window):
    suffix = '_anchored' if source == 'anchored' else ''
    fms = list(FM_TO_A)
    a = np.array([FM_TO_A[f] for f in fms])

    I_fm, W = [], None
    for fm in fms:
        path = os.path.join(data_dir, f'Real_data_projected_{fm}_no_exp{suffix}_v2.csv')
        P, W_f, M = load_samples(path, model)
        if W is not None and not np.allclose(W, W_f):
            raise ValueError(f'FM={fm}: omega grid differs from the other spacings')
        W = W_f
        I_fm.append(pair_I(M, P, W, window))
        print(f'FM={fm}: {M.shape[1]} bootstrap samples, P={list(P)}')
    n = min(len(I) for I in I_fm)
    I_fm = np.stack([I[:n] for I in I_fm])                      # (n_FM, n, n_W)
    print(f'using the first {n} samples of every FM')

    I0_a2, c2 = extrapolate(a, I_fm, 2)
    I0_a1, _ = extrapolate(a, I_fm, 1)

    I_mean = I0_a2.mean(axis=0)
    s_stat = I0_a2.std(axis=0, ddof=1)
    s_sys = np.abs(I_mean - I0_a1.mean(axis=0))
    s_tot = np.hypot(s_stat, s_sys)

    out = pd.DataFrame({'omega': W, 'I': I_mean, 'sigma_stat': s_stat,
                        'sigma_sys': s_sys, 'sigma_tot': s_tot,
                        'I_lin_a': I0_a1.mean(axis=0), 'c2': c2.mean(axis=0),
                        **{f'I_FM{f}': I_fm[k].mean(axis=0) for k, f in enumerate(fms)},
                        **{f'sigma_FM{f}': I_fm[k].std(axis=0, ddof=1)
                           for k, f in enumerate(fms)}})
    out_csv = os.path.join(data_dir, f'I_omega_continuum_boot_{model}{suffix}.csv')
    out.to_csv(out_csv, index=False)
    print(f'continuum I(omega) -> {out_csv}')
    i = int(np.argmax(I_mean))
    print(f'peak I = {I_mean[i]:.4f} +- {s_stat[i]:.4f} (stat) +- {s_sys[i]:.4f} (sys) '
          f'at omega = {W[i]:.1f}')

    plot(out, fms, a, model, source,
         os.path.join(data_dir, 'Images', f'I_omega_continuum_boot_{model}{suffix}'))


def plot(out, fms, a, model, source, out_base):
    w = out.omega.to_numpy()
    floor = lambda v: np.clip(v, 0.0, None)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=330)
    for k, (fm, av) in enumerate(zip(fms, a)):
        ax.plot(w, floor(out[f'I_FM{fm}']), STYLES[k % len(STYLES)], lw=1.2, alpha=0.85,
                color=COLORS[k % len(COLORS)], label=f'$a={av:.3f}$ fm (FM={fm})')
    ax.fill_between(w, floor(out.I - out.sigma_tot), floor(out.I + out.sigma_tot),
                    color='0.2', alpha=0.13, lw=0, label=r'stat $\oplus$ sys ($a$ vs $a^2$)')
    ax.fill_between(w, floor(out.I - out.sigma_stat), floor(out.I + out.sigma_stat),
                    color='0.2', alpha=0.32, lw=0, label='stat (bootstrap std)')
    ax.plot(w, floor(out.I), '-', lw=2.4, color='0.05',
            label=r'continuum $a\to0$: $I_0 + c\,a^{2}$')

    ax.set_xlim(w.min(), w.max())
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'Ioffe time $\omega$')
    ax.set_ylabel(r'$\Delta\mathcal{I}_g(\omega)$')
    ax.set_title(f'Continuum extrapolation of $I(\\omega)$ per bootstrap sample  '
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='../Data/2026')
    parser.add_argument("--source", choices=['roll', 'anchored'], default='roll',
                        help="roll: plain projection; anchored: reconstruct_real_data.py --anchor_real")
    parser.add_argument("--smooth_window", type=int, default=SMOOTH_WINDOW,
                        help="Savitzky-Golay window for I(omega), as in test_error.py")
    args = parser.parse_args()
    run(args.data_dir, best_model(args.data_dir), args.source, args.smooth_window)
