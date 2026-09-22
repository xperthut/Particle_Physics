"""Post-process the reconstructed real-data M(omega) curves into I(omega).

Converted from test_error.ipynb and fitted to the current 2026 layout.

reconstruct_real_data.py writes one file per (FM, model variant):
    Data/2026/Real_data_projected_{fm}_no_exp_v2.csv     (rolled with *_exp_No)
    Data/2026/Real_data_projected_{fm}_with_exp_v2.csv    (rolled with *_exp_Yes)
each with columns  FM, P, Exp, W, anchored, M_GB, M_RF, M_XGB.  This script
processes every such file it finds and, per (variant, FM):

  * averages each model's M over the Exp replicas -> Mean_M_* / Std_M_* per (P, W);
  * combines two momenta into the Ioffe-time distribution
        I(w) = (p2^2 * M(p2, w) - p1^2 * M(p1, w)) / (p2^2 - p1^2)
    on the Exp-averaged mean curve.

    The pair is referenced to P=1 (default p1=1, p2=2): this is the exact
    inverse of how synthetic_data_generator.py *defines* I(w),
        I = (4*M2 - M1) / 3          (A=-1/3, B=4/3 in that file)
    and generally  I = (k^2 * M_k - M1) / (k^2 - 1)  for the derived moments.
    A pair that excludes P=1 (e.g. 2,3) is inconsistent with that definition
    and gives a sign-flipped curve.

  * smooths I(w) with a Savitzky-Golay filter (order-3 polynomial over a
    sliding window; --smooth_window points, default 25 = 2.5 in omega).
    Savitzky-Golay keeps the single hump / peak position that a plain moving
    average (the old seasonal_decompose trend) rounded off, while removing the
    step artefacts the autoregressive roll leaves in the extrapolated tail.

  * writes, per (variant, FM):
        Data/2026/Projected_M_mean_std_{variant}_{fm}.csv
        Data/2026/Projected_I_{variant}_{fm}.csv
        Data/2026/Images/projected_I_{variant}_{fm}.png

Run from Helicity/Code:  python test_error.py
"""

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")            # script context: render to files, no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

import warnings
warnings.filterwarnings("ignore")

pd.options.mode.chained_assignment = None

MODEL_NAMES = ['GB', 'RF', 'XGB']
MODEL_TITLES = {'GB': 'Gradient boosted tree',
                'RF': 'Random forest',
                'XGB': 'Extreme gradient boosted tree'}
VARIANT_TITLE = {'no_exp': 'without Exp feature', 'with_exp': 'with Exp feature'}
W_DECIMALS = 1
SMOOTH_WINDOW = 60       # Savitzky-Golay window in grid points (0.1 each)
SMOOTH_POLYORDER = 3

_FILE_RE = re.compile(r'Real_data_projected_(.+)_(no_exp|with_exp)_v2\.csv$')


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


def get_mean_std(data, grp_cols, agg_col):
    """Mean_{agg_col} / Std_{agg_col} of ``agg_col`` grouped by ``grp_cols``."""
    g = data.groupby(grp_cols)[agg_col]
    df_mu = g.mean().reset_index().rename(columns={agg_col: f'Mean_{agg_col}'})
    df_sigma = g.std(ddof=0).reset_index().rename(columns={agg_col: f'Std_{agg_col}'})
    return df_mu.merge(df_sigma, on=grp_cols)


class Test_Error:
    def __init__(self, data_dir='../Data/2026', p1=1, p2=2, smooth_window=SMOOTH_WINDOW):
        self.data_dir = data_dir
        self.p1 = p1
        self.p2 = p2
        self.smooth_window = smooth_window
        self.img_dir = os.path.join(data_dir, 'Images')

    def _discover(self):
        """[(filename, fm, variant), ...] for every projected file present."""
        out = []
        for name in sorted(os.listdir(self.data_dir)):
            m = _FILE_RE.match(name)
            if m:
                out.append((name, m.group(1), m.group(2)))
        return out

    def _mean_std_curves(self, data):
        """One frame indexed by (P, W) with Mean_M_* / Std_M_* per model."""
        df = None
        for name in MODEL_NAMES:
            part = get_mean_std(data, ['P', 'W'], f'M_{name}')
            df = part if df is None else df.merge(part, on=['P', 'W'])
        return df

    def _compute_I(self, df_ms):
        """I(w) + smoothed I(w) from the (p1, p2) momentum pair."""
        p1, p2 = self.p1, self.p2
        present = set(df_ms.P.unique())
        if not {p1, p2} <= present:
            print(f'  P pair ({p1}, {p2}) not both present in {sorted(present)}; '
                  f'skipping I(omega)')
            return None

        denom = p2 ** 2 - p1 ** 2
        rows = {}
        piv = None
        for name in MODEL_NAMES:
            piv = (df_ms.pivot_table(index='W', columns='P', values=f'Mean_M_{name}')
                   .sort_index())
            rows[f'I_{name}'] = ((p2 ** 2 * piv[p2] - p1 ** 2 * piv[p1]) / denom).to_numpy()
        rows['W'] = np.round(piv.index.to_numpy(), W_DECIMALS)

        df_I = pd.DataFrame(rows).sort_values('W').reset_index(drop=True)
        for name in MODEL_NAMES:
            sm = get_smooth_curve(df_I[f'I_{name}'], self.smooth_window)
            sm = np.where(df_I['W'].to_numpy() == 0.0, 0.0, sm)   # I(0) = 0 exactly
            df_I[f'smooth_I{name}'] = sm
        return df_I[['W'] + [f'I_{n}' for n in MODEL_NAMES]
                    + [f'smooth_I{n}' for n in MODEL_NAMES]]

    def _plot(self, fm, variant, df_ms, df_I, w_anchor):
        ordered_ps = sorted(df_ms.P.unique())
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        n_rows = 2 if df_I is not None else 1
        fig, axes = plt.subplots(n_rows, 3, figsize=(24, 9 * n_rows), dpi=150,
                                 squeeze=False)

        for col, name in enumerate(MODEL_NAMES):
            ax = axes[0, col]
            f_mu, f_sd = f'Mean_M_{name}', f'Std_M_{name}'
            for k, p in enumerate(ordered_ps):
                t = df_ms[df_ms.P == p]
                c = colors[k % len(colors)]
                ax.errorbar(t.W, t[f_mu], yerr=t[f_sd], ecolor=c, color=c,
                            label=f'$p_z$={p}' if col == 0 else None)
            ax.set_title(MODEL_TITLES[name])
            ax.set_xlabel(r'$\omega$')
            if col == 0:
                ax.set_ylabel(r'$\Delta\mathfrak{M}(\omega)$')
                ax.legend()

            if df_I is not None:
                axI = axes[1, col]
                axI.plot(df_I['W'], df_I[f'I_{name}'], '.', ms=4, alpha=0.35,
                         color=colors[0], label='raw')
                axI.plot(df_I['W'], df_I[f'smooth_I{name}'], '-', lw=2,
                         color=colors[3], label='smoothed')
                if w_anchor and w_anchor > 0:
                    axI.axvline(w_anchor, ls='--', lw=1, color='0.5')
                axI.set_xlabel(r'$\omega$')
                if col == 0:
                    axI.set_ylabel(r'$\Delta\mathcal{I}_g(\omega)$')
                    axI.legend()

        anchor_note = (f'   (real data anchors $\\omega\\leq${w_anchor:.1f}, '
                       f'dashed line; beyond is extrapolation)' if w_anchor else '')
        fig.suptitle(f'FM={fm}  [{VARIANT_TITLE.get(variant, variant)}]   '
                     f'I($\\omega$) from $p_z$=({self.p1}, {self.p2}){anchor_note}')
        fig.tight_layout()
        os.makedirs(self.img_dir, exist_ok=True)
        out_png = os.path.join(self.img_dir, f'projected_I_{variant}_{fm}.png')
        fig.savefig(out_png)
        plt.close(fig)
        print(f'  plot -> {out_png}')

    def run(self):
        found = self._discover()
        if not found:
            print(f'no Real_data_projected_*_{{no_exp,with_exp}}_v2.csv in {self.data_dir}')
            return

        for name, fm, variant in found:
            data = pd.read_csv(os.path.join(self.data_dir, name),
                               usecols=lambda c: c != 'FM')
            data['W'] = np.round(data['W'], W_DECIMALS)
            print(f'[{variant}] FM={fm}: {len(data)} rows, P={sorted(data.P.unique())}, '
                  f'{data.Exp.nunique()} Exp replicas')

            df_ms = self._mean_std_curves(data)
            df_I = self._compute_I(df_ms)

            w_anchor = 0.0
            if 'anchored' in data.columns:
                anch = data[(data.anchored == 1) & data.P.isin([self.p1, self.p2])]
                w_anchor = float(anch.W.max()) if not anch.empty else 0.0

            ms_out = os.path.join(self.data_dir, f'Projected_M_mean_std_{variant}_{fm}.csv')
            df_ms.to_csv(ms_out, index=False)
            print(f'  mean/std -> {ms_out}')

            if df_I is not None:
                I_out = os.path.join(self.data_dir, f'Projected_I_{variant}_{fm}.csv')
                df_I.to_csv(I_out, index=False)
                print(f'  I(omega) -> {I_out}')

            self._plot(fm, variant, df_ms, df_I, w_anchor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='../Data/2026',
                        help="Directory holding Real_data_projected_{fm}_{variant}_v2.csv")
    parser.add_argument("--p1", type=int, default=1,
                        help="reference momentum of the I(omega) pair (keep at 1)")
    parser.add_argument("--p2", type=int, default=2,
                        help="second momentum of the I(omega) pair (2 = exact; k>=3 has gamma damping)")
    parser.add_argument("--smooth_window", type=int, default=SMOOTH_WINDOW,
                        help="Savitzky-Golay window in omega-grid points (odd; larger = smoother)")
    args = parser.parse_args()

    Test_Error(data_dir=args.data_dir, p1=args.p1, p2=args.p2,
               smooth_window=args.smooth_window).run()
