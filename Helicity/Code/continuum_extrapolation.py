"""Continuum extrapolation (a -> 0) of the best model's I(omega).

Reads the long-format file written by test_error.py
    Data/2026/best_model_I_no_exp_{model}.csv      (columns FM, W, I)
where FM is the lattice spacing in units of 0.01 fm (9, 12, 15 -> a = 0.09,
0.12, 0.15 fm).  At each fixed omega the per-a values are fit by least squares to

    I(omega, a) = I(omega, 0) + c2(omega) * a**2

the leading cutoff term for an O(a)-improved action.  A second fit, linear in
a, is kept only as a systematic band.  The file carries no pointwise errors, so
that band reflects the choice of cutoff dependence, not a statistical error.
With three spacings each fit has one degree of freedom; RMS_a2 / RMS_a report
how well each form describes the three points.

Writes
    Data/2026/I_omega_continuum.csv
        omega, I_a0_a2, c2, RMS_a2, I_a0_a, c1, RMS_a, I_a<spacing> per FM
    Data/2026/Images/I_omega_continuum.{png,pdf}

The CSV keeps the unclipped fit values; only the plot floors I at 0 (I(omega)
is physically non-negative).

Run from Helicity/Code:  python continuum_extrapolation.py
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")            # script context: render to files, no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

FM_TO_A = 0.01                   # FM column is a in units of 0.01 fm
W_DECIMALS = 1
# validated categorical slots 1-3 (same as test_error.py's best-model plot),
# with a line style each so the spacings are not told apart by colour alone
COLORS = ['#2a78d6', '#eb6834', '#1baf7a']
STYLES = ['-', '--', '-.']


def fit_cutoff(a, I, power):
    """Least-squares fit I = I0 + c * a**power at every omega.

    a: (n_a,) spacings; I: (n_omega, n_a).  Returns I0, c, rms residual, each (n_omega,).
    """
    X = np.column_stack([np.ones_like(a), a ** power])
    coef, *_ = np.linalg.lstsq(X, I.T, rcond=None)          # (2, n_omega)
    resid = I - (X @ coef).T
    return coef[0], coef[1], np.sqrt(np.mean(resid ** 2, axis=1))


def load_curves(src):
    """Pivot the long FM, W, I file into (omega, a, I[omega, a])."""
    df = pd.read_csv(src)
    df['W'] = np.round(df['W'], W_DECIMALS)
    piv = df.pivot_table(index='W', columns='FM', values='I').sort_index()
    piv = piv.reindex(columns=sorted(piv.columns))
    missing = piv.isna().any(axis=1)
    if missing.any():
        print(f'dropping {int(missing.sum())} omega points not present for every FM')
        piv = piv[~missing]
    if piv.shape[1] < 3:
        raise ValueError(f'need at least 3 lattice spacings, got FM={list(piv.columns)}')
    return piv.index.to_numpy(), piv.columns.to_numpy(), piv.to_numpy()


def plot(omega, fms, a, I, I0_a2, I0_a, out_base):
    fig, ax = plt.subplots(figsize=(9, 6), dpi=330)
    for k, (fm, av) in enumerate(zip(fms, a)):
        ax.plot(omega, I[:, k], STYLES[k % len(STYLES)], lw=1.6,
                color=COLORS[k % len(COLORS)], label=f'$a={av:.2f}$ fm (FM={fm})')

    lo = np.clip(np.minimum(I0_a2, I0_a), 0.0, None)
    hi = np.clip(np.maximum(I0_a2, I0_a), 0.0, None)
    ax.fill_between(omega, lo, hi, color='0.55', alpha=0.25, lw=0,
                    label=r'$a$ vs $a^{2}$ systematic band')
    ax.plot(omega, np.clip(I0_a2, 0.0, None), '-', lw=2.4, color='0.1',
            label=r'continuum $a\to 0$ (fit $I_0 + c_2 a^{2}$)')

    ax.set_xlim(omega.min(), omega.max())
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'Ioffe time $\omega$')
    ax.set_ylabel(r'$\Delta\mathcal{I}_g(\omega)$')
    ax.set_title(r'Continuum extrapolation of $I(\omega)$')
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(alpha=0.25, lw=0.5)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out_base}.{ext}', bbox_inches='tight')
        print(f'plot -> {out_base}.{ext}')
    plt.close(fig)


def main(src, data_dir):
    omega, fms, I = load_curves(src)
    a = fms.astype(float) * FM_TO_A
    print(f'{src}: {len(omega)} omega points, a = {list(a)} fm')

    I0_a2, c2, rms_a2 = fit_cutoff(a, I, 2)
    I0_a, c1, rms_a = fit_cutoff(a, I, 1)

    out = pd.DataFrame({'omega': omega,
                        'I_a0_a2': I0_a2, 'c2': c2, 'RMS_a2': rms_a2,
                        'I_a0_a': I0_a, 'c1': c1, 'RMS_a': rms_a,
                        **{f'I_a{av:.2f}': I[:, k] for k, av in enumerate(a)}})
    out_csv = os.path.join(data_dir, 'I_omega_continuum.csv')
    out.to_csv(out_csv, index=False)
    print(f'continuum data -> {out_csv}')
    print(f'max RMS residual: a^2 fit {rms_a2.max():.3e}, linear-a fit {rms_a.max():.3e}')

    plot(omega, fms, a, I, I0_a2, I0_a,
         os.path.join(data_dir, 'Images', 'I_omega_continuum'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default='../Data/2026/best_model_I_no_exp_GP.csv',
                        help="Long-format FM, W, I file (from test_error.py)")
    parser.add_argument("--data_dir", default='../Data/2026',
                        help="Where to write I_omega_continuum.csv and Images/")
    args = parser.parse_args()
    main(args.src, args.data_dir)
