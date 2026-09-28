"""Collect one model's I(omega) for every FM into a single frame and plot them together.

Reads the per-FM outputs of test_error.py
    Data/2026/Projected_I_{variant}_{fm}.csv
and takes that model's smoothed curve (smooth_I{model}) as I_omega.  Writes
    Data/2026/I_omega_all_FM_{model}_{variant}.csv      columns FM, omega, I_omega
    Data/2026/Images/I_omega_all_FM_{model}_{variant}.png

Each curve is drawn solid over the omega range covered by that FM's real
lattice data (All_p_w_m.csv) and dashed beyond it, where it is pure
extrapolation.

Run from Helicity/Code (after test_error.py):  python combine_I_omega.py
"""

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")            # script context: render to files, no display
import matplotlib.pyplot as plt
import pandas as pd

# categorical slots 1-3 of the dataviz reference palette, in fixed order
FM_COLORS = ['#2a78d6', '#eb6834', '#1baf7a']
FM_MARKERS = ['o', 's', '^']     # secondary encoding so identity is not color-alone


def collect(data_dir, model, variant):
    """FM, omega, I_omega for every Projected_I_{variant}_{fm}.csv present."""
    file_re = re.compile(rf'Projected_I_{variant}_(.+)\.csv$')
    parts = []
    for name in sorted(os.listdir(data_dir)):
        m = file_re.match(name)
        if not m:
            continue
        df = pd.read_csv(os.path.join(data_dir, name), usecols=['W', f'smooth_I{model}'])
        parts.append(pd.DataFrame({'FM': int(m.group(1)),
                                   'omega': df['W'],
                                   'I_omega': df[f'smooth_I{model}']}))
    if not parts:
        raise SystemExit(f'no Projected_I_{variant}_*.csv in {data_dir}; run test_error.py first')
    return pd.concat(parts).sort_values(['FM', 'omega']).reset_index(drop=True)


def plot(df, w_real, model, variant, out_png):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    for k, (fm, g) in enumerate(df.groupby('FM')):
        c, mk = FM_COLORS[k % len(FM_COLORS)], FM_MARKERS[k % len(FM_MARKERS)]
        w_max = w_real.get(fm)
        inside = g if w_max is None else g[g.omega <= w_max]
        beyond = g if w_max is None else g[g.omega >= w_max]
        ax.plot(inside.omega, inside.I_omega, '-', lw=2, color=c,
                marker=mk, markevery=10, ms=8, label=f'FM = {fm:02d}')
        if w_max is not None:
            ax.plot(beyond.omega, beyond.I_omega, '--', lw=2, color=c,
                    marker=mk, markevery=10, ms=8)

    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel(r'$\Delta\mathcal{I}_g(\omega)$')
    ax.set_title(f'{model} reconstruction ({variant}): solid = within real-data '
                 r'$\omega$ range, dashed = extrapolation', fontsize=10)
    ax.grid(True, color='0.9', lw=0.8)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(title='Ensemble', frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='../Data/2026')
    parser.add_argument("--model", default='GB', choices=['GB', 'RF', 'XGB'])
    parser.add_argument("--variant", default='no_exp', choices=['no_exp', 'with_exp'])
    args = parser.parse_args()

    df = collect(args.data_dir, args.model, args.variant)
    tag = f'{args.model}_{args.variant}'
    out_csv = os.path.join(args.data_dir, f'I_omega_all_FM_{tag}.csv')
    df.to_csv(out_csv, index=False)
    print(f'{len(df)} rows, FM={sorted(df.FM.unique())} -> {out_csv}')

    real = pd.read_csv(os.path.join(args.data_dir, 'All_p_w_m.csv'), usecols=['FM', 'W'])
    w_real = real.groupby('FM').W.max().to_dict()

    out_png = os.path.join(args.data_dir, 'Images', f'I_omega_all_FM_{tag}.png')
    plot(df, w_real, args.model, args.variant, out_png)
    print(f'plot -> {out_png}')
