"""Post-process the reconstructed real-data M(omega) curves into I(omega).

Converted from test_error.ipynb and fitted to the current 2026 layout.

reconstruct_real_data.py writes one file per (FM, model variant):
    Data/2026/Real_data_projected_{fm}_no_exp_v2.csv     (rolled with *_exp_No)
    Data/2026/Real_data_projected_{fm}_with_exp_v2.csv    (rolled with *_exp_Yes)
each with columns  FM, P, Exp, W, anchored, M_<model> for every model that was
rolled (GB, RF, XGB, NN, GP -- whichever were trained).  This script
processes only the no_exp files (VARIANT) and, per FM:

  * averages each model's M over the Exp replicas -> Mean_M_* / Std_M_* per (P, W);
  * combines every momentum pair (2 <= p1 < p2; P=1 is left out of the pairs only --
    it is still averaged, written and plotted as an M curve) into an Ioffe-time estimate
        I(w) = (p2^2 * M(p2, w) - p1^2 * M(p1, w)) / (p2^2 - p1^2)
    on the Exp-averaged mean curve, then averages the non-negative pair
    estimates at each w (0 where every pair is negative).  With gamma=0,
    synthetic_data_generator.py derives every M_k so that any pair inverts
    to the same I(w).

  * smooths I(w) with a Savitzky-Golay filter (order-3 polynomial over a
    sliding window; --smooth_window points, default 60 -> 59 = 5.9 in omega).
    Savitzky-Golay keeps the single hump / peak position that a plain moving
    average (the old seasonal_decompose trend) rounded off, while removing the
    step artefacts the autoregressive roll leaves in the extrapolated tail.

  * writes, per (variant, FM):
        Data/2026/Projected_M_mean_std_{variant}_{fm}.csv
        Data/2026/Projected_I_{variant}_{fm}.csv
        Data/2026/Images/projected_I_{variant}_{fm}.png

  * test error: scores each model's projection against the real lattice M in
    Data/2026/All_p_w_m.csv at every real (P, Exp, W) point past the
    N_SEED-point seed window -- points the autoregressive roll never saw.  The
    projected curve (0.1 omega grid) is linearly interpolated onto the real,
    unrounded omega and compared replica-by-replica (same Exp), giving RMSE / R2
    per (FM, model), plus RMSE_mean on the Exp-averaged curves.  Models are
    ranked overall by the average of their rank in mean RMSE and in mean R2
    across FMs, written to
        Data/2026/test_error_summary_no_exp.csv    (per FM, per model)
        Data/2026/test_error_ranking_no_exp.csv    (overall ranking)
    and only the top three are plotted (M with the real data overlaid, and I).
    The single best model's smoothed I(w) for every FM is also drawn on one axis:
        Data/2026/Images/best_model_I_no_exp_{model}.png
        Data/2026/best_model_I_no_exp_{model}.csv     (FM, W, smoothed I)

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
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

pd.options.mode.chained_assignment = None

MODEL_NAMES = ['GB', 'RF', 'XGB', 'NN', 'GP']
MODEL_TITLES = {'GB': 'Gradient boosted tree',
                'RF': 'Random forest',
                'XGB': 'Extreme gradient boosted tree',
                'NN': 'Neural network (MLP)',
                'GP': 'Gaussian process'}
VARIANT_TITLE = {'no_exp': 'without Exp feature', 'with_exp': 'with Exp feature'}
W_DECIMALS = 1
SMOOTH_WINDOW = 60       # Savitzky-Golay window in grid points (0.1 each)
SMOOTH_POLYORDER = 3
MIN_P = 2                # smallest P allowed in an I(omega) pair; M curves keep every P (incl. P=1)
N_SEED = 4               # seed-window points per P in reconstruct_real_data.py (N_LAGS)
TOP_K = 3
# Only the *_exp_No models are evaluated: Exp is an arbitrary replica label, so
# feeding it as a feature only invites per-replica memorisation.
VARIANT = 'no_exp'

_FILE_RE = re.compile(r'Real_data_projected_(.+)_(no_exp|with_exp)_v2\.csv$')


def get_smooth_curve(val, window=SMOOTH_WINDOW, polyorder=SMOOTH_POLYORDER):
    """Savitzky-Golay smooth of a 1-D series on a uniform grid.

    The window is forced odd and shrunk to fit short inputs; if it cannot be
    made larger than ``polyorder`` the series is returned unchanged.
    """
    val = np.asarray(val, dtype=float)
    n = val.size
    w = min(int(window), n)
    if w % 2 == 0:
        w -= 1
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
    def __init__(self, data_dir='../Data/2026', smooth_window=SMOOTH_WINDOW, top_k=TOP_K):
        self.data_dir = data_dir
        self.top_k = top_k
        self.smooth_window = smooth_window
        self.img_dir = os.path.join(data_dir, 'Images')

    def _discover(self):
        """[(filename, fm, variant), ...] for every projected file present."""
        out = []
        for name in sorted(os.listdir(self.data_dir)):
            m = _FILE_RE.match(name)
            if m and m.group(2) == VARIANT:
                out.append((name, m.group(1), m.group(2)))
        return out

    def _mean_std_curves(self, data, names):
        """One frame indexed by (P, W) with Mean_M_* / Std_M_* per model."""
        df = None
        for name in names:
            part = get_mean_std(data, ['P', 'W'], f'M_{name}')
            df = part if df is None else df.merge(part, on=['P', 'W'])
        return df

    @staticmethod
    def _I_omega(piv, p1, p2):
        """('p1-p2', I(w) array) from one momentum pair of the pivot (index W, columns P)."""
        denom = p2 ** 2 - p1 ** 2
        col = f'{p1}-{p2}'
        vals = ((p2 ** 2 * piv[p2] - p1 ** 2 * piv[p1]) / denom).to_numpy()
        return col, vals

    def _compute_I(self, df_ms, names):
        """I(w) averaged over every (MIN_P <= i < j) momentum pair, + smoothed I(w).

        At each W only the non-negative pair estimates enter the mean; if every
        pair is negative there, I(w) is set to 0 (I is physically non-negative).
        """
        present = sorted(p for p in df_ms.P.unique() if p >= MIN_P)
        if len(present) < 2:
            print(f'  need at least two P >= {MIN_P}, have {present}; skipping I(omega)')
            return None

        rows = {}
        piv = None
        for name in names:
            piv = (df_ms.pivot_table(index='W', columns='P', values=f'Mean_M_{name}')
                   .sort_index())
            tmp = {}
            for i in present:
                for j in present:
                    if i < j:
                        c, v = self._I_omega(piv, i, j)
                        tmp[c] = v
            tmp = pd.DataFrame(tmp, index=piv.index)
            rows[f'I_{name}'] = tmp.where(tmp >= 0).mean(axis=1).fillna(0.0).to_numpy()
        rows['W'] = np.round(piv.index.to_numpy(), W_DECIMALS)

        df_I = pd.DataFrame(rows).sort_values('W').reset_index(drop=True)
        for name in names:
            sm = get_smooth_curve(df_I[f'I_{name}'], self.smooth_window)
            sm = np.where(df_I['W'].to_numpy() == 0.0, 0.0, sm)   # I(0) = 0 exactly
            # I(omega) is physically non-negative everywhere; the raw combination
            # can dip slightly below 0 at small omega (real M is noisiest there,
            # e.g. one anchored point sized to fit its own P but not the other's),
            # and the Savitzky-Golay window then smears that dip into a visible
            # negative glitch just past omega=0. Floor the *smoothed* curve only
            # -- the raw I_* columns keep the unclipped values for diagnostics.
            sm = np.clip(sm, 0.0, None)
            df_I[f'smooth_I{name}'] = sm
        return df_I[['W'] + [f'I_{n}' for n in names]
                    + [f'smooth_I{n}' for n in names]]

    def _plot(self, fm, variant, df_ms, df_I, w_anchor, names, df_real=None, overall=None):
        """M(w) (top row) and I(w) (bottom row), one column per model in ``names``.

        Real lattice M (Exp mean +- std) is overlaid on the M panels as open
        markers, and each column title carries the model's overall test rank.
        """
        ordered_ps = sorted(df_ms.P.unique())
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        real_ms = None
        if df_real is not None and not df_real.empty:
            real_ms = get_mean_std(df_real, ['P', 'W'], 'M')
        n_rows = 2 if df_I is not None else 1
        fig, axes = plt.subplots(n_rows, len(names), figsize=(8 * len(names), 9 * n_rows),
                                 dpi=330, squeeze=False)

        for col, name in enumerate(names):
            ax = axes[0, col]
            f_mu, f_sd = f'Mean_M_{name}', f'Std_M_{name}'
            for k, p in enumerate(ordered_ps):
                t = df_ms[df_ms.P == p]
                c = colors[k % len(colors)]
                ax.errorbar(t.W, t[f_mu], yerr=t[f_sd], ecolor=c, color=c,
                            label=f'$p_z$={p}' if col == 0 else None)
                if real_ms is not None:
                    r = real_ms[real_ms.P == p]
                    ax.errorbar(r.W, r.Mean_M, yerr=r.Std_M, fmt='o', mfc='none', ms=7,
                                color=c, label='real data' if col == 0 and k == 0 else None)
            title = MODEL_TITLES[name]
            if overall is not None and name in overall.index:
                o = overall.loc[name]
                title = (f'#{int(o.overall_rank)}  {title}\n'
                         f'test RMSE={o.mean_RMSE:.4f}, $R^2$={o.mean_R2:.3f} (mean over FMs)')
            ax.set_title(title)
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
        top_note = f'top {len(names)} models by test error   ' if overall is not None else ''
        fig.suptitle(f'FM={fm}  [{VARIANT_TITLE.get(variant, variant)}]   {top_note}'
                     f'I($\\omega$) averaged over all $p_z\\geq${MIN_P} pairs{anchor_note}')
        fig.tight_layout(rect=(0, 0, 1, 0.97))   # leave room for the suptitle
        os.makedirs(self.img_dir, exist_ok=True)
        out_png = os.path.join(self.img_dir, f'projected_I_{variant}_{fm}.png')
        fig.savefig(out_png, dpi=330)
        plt.close(fig)
        print(f'  plot -> {out_png}')

    def _plot_best(self, best, per_fm, overall):
        """Smoothed I(w) of the single best model, every FM on one axis."""
        colors = ['#2a78d6', '#eb6834', '#1baf7a']     # validated categorical slots 1-3
        styles = ['-', '--', '-.']                      # secondary encoding (not colour alone)
        fig, ax = plt.subplots(figsize=(9, 6), dpi=330)
        w_anchor = 0.0
        curves = []                                     # plotted data, long format FM, W, I
        fms = sorted(per_fm, key=lambda t: float(t[0]))
        for k, (fm, _variant, _ms, df_I, w_anc, _names, _real) in enumerate(fms):
            if df_I is None or f'smooth_I{best}' not in df_I:
                continue
            w, y = df_I['W'].to_numpy(), df_I[f'smooth_I{best}'].to_numpy()
            c = colors[k % len(colors)]
            ax.plot(w, y, styles[k % len(styles)], lw=2, color=c, label=f'FM={fm}')
            i = int(np.argmax(y))                       # direct label at the peak
            ax.annotate(f'FM={fm}', (w[i], y[i]), xytext=(0, 6), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9, color='0.2')
            w_anchor = max(w_anchor, w_anc)
            curves.append(pd.DataFrame({'FM': fm, 'W': w, 'I': y}))
        if curves:
            out_csv = os.path.join(self.data_dir, f'best_model_I_{VARIANT}_{best}.csv')
            pd.concat(curves, ignore_index=True).to_csv(out_csv, index=False)
            print(f'  best-model data -> {out_csv}')
        if w_anchor > 0:
            ax.axvline(w_anchor, ls=':', lw=1, color='0.5')
        ax.grid(alpha=0.25, lw=0.5)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.set_xlabel(r'$\omega$')
        ax.set_ylabel(r'$\Delta\mathcal{I}_g(\omega)$')
        o = overall.loc[best]
        ax.set_title(f'{MODEL_TITLES[best]} (best mean test error: RMSE={o.mean_RMSE:.4f}, '
                     f'$R^2$={o.mean_R2:.3f})\n'
                     f'smoothed I($\\omega$), {VARIANT_TITLE[VARIANT]}', fontsize=10)
        ax.legend(frameon=False)
        fig.tight_layout()
        out_png = os.path.join(self.img_dir, f'best_model_I_{VARIANT}_{best}.png')
        fig.savefig(out_png, dpi=330)
        plt.close(fig)
        print(f'  best-model plot -> {out_png}')

    def _load_real(self):
        path = os.path.join(self.data_dir, 'All_p_w_m.csv')
        if not os.path.exists(path):
            print(f'{path} not found; test error skipped')
            return None
        return pd.read_csv(path, usecols=['FM', 'P', 'Exp', 'W', 'M'])

    @staticmethod
    def _test_error(data, df_real, names):
        """{model: (rmse, r2, rmse_mean, n)} of the projection vs the real M.

        Test points are the real (P, Exp, W) past the seed window (the first
        N_SEED grid omegas per P, which are copied from real data / synthetic
        mean). Each model's per-Exp curve is linearly interpolated onto the
        real omega and compared to the same Exp's real M.
        """
        y_true, y_pred, keys = [], {n: [] for n in names}, []
        for p in sorted(data.P.unique()):
            proj = data[data.P == p]
            real = df_real[df_real.P == p]
            grid = np.sort(proj.W.unique())
            if real.empty or len(grid) <= N_SEED:
                continue
            real = real[(real.W > grid[N_SEED - 1] + 1e-9) & (real.W <= grid[-1])
                        & real.Exp.isin(proj.Exp.unique())]
            if real.empty:
                continue

            w = real.W.to_numpy()
            idx = np.clip(np.searchsorted(grid, w, side='right') - 1, 0, len(grid) - 2)
            frac = (w - grid[idx]) / (grid[idx + 1] - grid[idx])
            rows = np.arange(len(real))
            for n in names:
                piv = (proj.pivot_table(index='Exp', columns='W', values=f'M_{n}')
                       .reindex(columns=grid))
                arr = piv.reindex(real.Exp.to_numpy()).to_numpy()   # one row per real point
                y_pred[n].append(arr[rows, idx] * (1 - frac) + arr[rows, idx + 1] * frac)
            y_true.append(real.M.to_numpy())
            keys.append(real[['P', 'W']])

        if not y_true:
            return {}
        y_true = np.concatenate(y_true)
        keys = pd.concat(keys, ignore_index=True)
        out = {}
        for n in names:
            pred = np.concatenate(y_pred[n])
            ok = np.isfinite(pred)
            # Exp-averaged curves: mean real M vs mean prediction per (P, W)
            mean = (keys.assign(t=y_true, y=pred)[ok].groupby(['P', 'W'])[['t', 'y']].mean())
            out[n] = (float(np.sqrt(mean_squared_error(y_true[ok], pred[ok]))),
                      float(r2_score(y_true[ok], pred[ok])),
                      float(np.sqrt(mean_squared_error(mean.t, mean.y))),
                      int(ok.sum()))
        return out

    def _rank(self, rows):
        """Write the per-FM summary and the overall ranking; return the TOP_K model names.

        Within one FM, RMSE and R2 share the same y_true, so they order the
        models identically.  Across FMs they can disagree, so the overall score
        is the average of the model's rank by mean RMSE (low = good) and by
        mean R2 (high = good), ties broken by mean RMSE.
        """
        df = pd.DataFrame(rows)
        df['rank'] = df.groupby('FM')['RMSE'].rank(method='min').astype(int)
        df = df.sort_values(['FM', 'rank']).reset_index(drop=True)
        out = os.path.join(self.data_dir, f'test_error_summary_{VARIANT}.csv')
        df.to_csv(out, index=False)
        print(f'\ntest error summary -> {out}')
        print(df.to_string(index=False))

        overall = (df.groupby('model')
                   .agg(mean_RMSE=('RMSE', 'mean'), mean_R2=('R2', 'mean'),
                        mean_RMSE_mean=('RMSE_mean', 'mean'), n_FM=('FM', 'nunique'))
                   .reset_index())
        overall['rank_RMSE'] = overall.mean_RMSE.rank(method='min')
        overall['rank_R2'] = overall.mean_R2.rank(method='min', ascending=False)
        overall['score'] = (overall.rank_RMSE + overall.rank_R2) / 2
        overall = overall.sort_values(['score', 'mean_RMSE']).reset_index(drop=True)
        overall['overall_rank'] = np.arange(1, len(overall) + 1)
        out = os.path.join(self.data_dir, f'test_error_ranking_{VARIANT}.csv')
        overall.to_csv(out, index=False)
        print(f'\noverall ranking (mean RMSE + mean R2 across FMs) -> {out}')
        print(overall.to_string(index=False))

        self._write_train_test(df, overall)

        top = overall.model.head(self.top_k).tolist()
        print(f'\ntop {self.top_k} models: {top}')
        return top, overall.set_index('model')

    def _write_train_test(self, df_test, overall):
        """Train (train_error.py) and test error side by side, per FM and overall.

        Uses the matching train rows (with_exp = No for the no_exp variant) from
        train_error_summary.csv; skipped with a note if that file is missing.
        Writes Data/2026/train_test_error_{VARIANT}.csv: one row per (FM, model)
        plus FM='all' rows with the means across FMs, ordered by overall rank.
        """
        path = os.path.join(self.data_dir, 'train_error_summary.csv')
        if not os.path.exists(path):
            print(f'{path} not found -- run train_error.py first; train/test csv skipped')
            return
        exp_id = 'Yes' if VARIANT == 'with_exp' else 'No'
        tr = pd.read_csv(path)
        tr = (tr[tr.with_exp == exp_id]
              .rename(columns={'n_rows': 'train_n', 'RMSE': 'train_RMSE', 'R2': 'train_R2'})
              [['FM', 'model', 'train_n', 'train_RMSE', 'train_R2']])
        te = (df_test.rename(columns={'n_points': 'test_n', 'RMSE': 'test_RMSE',
                                      'R2': 'test_R2', 'RMSE_mean': 'test_RMSE_mean',
                                      'rank': 'test_rank_in_FM'})
              [['FM', 'model', 'test_n', 'test_RMSE', 'test_R2', 'test_RMSE_mean',
                'test_rank_in_FM']])
        te['FM'] = te.FM.astype(str)
        tr['FM'] = tr.FM.astype(str)
        per_fm = te.merge(tr, on=['FM', 'model'], how='left')

        means = (per_fm.groupby('model')[['train_RMSE', 'train_R2', 'test_RMSE', 'test_R2',
                                          'test_RMSE_mean']].mean().reset_index()
                 .assign(FM='all'))
        rank = overall.set_index('model').overall_rank
        out_df = pd.concat([per_fm, means], ignore_index=True)
        out_df.insert(0, 'overall_rank', out_df.model.map(rank).astype(int))
        out_df['_all'] = out_df.FM == 'all'
        out_df = (out_df.sort_values(['_all', 'FM', 'overall_rank'], key=lambda s: s.map(
                      lambda v: int(v) if isinstance(v, str) and v.isdigit() else v))
                  .drop(columns='_all').reset_index(drop=True))
        out = os.path.join(self.data_dir, f'train_test_error_{VARIANT}.csv')
        out_df.to_csv(out, index=False)
        print(f'\ntrain + test error -> {out}')
        print(out_df.to_string(index=False, float_format=lambda x: f'{x:.5g}'))

    def run(self):
        found = self._discover()
        if not found:
            print(f'no Real_data_projected_*_{VARIANT}_v2.csv in {self.data_dir}')
            return

        df_real = self._load_real()
        err_rows, per_fm = [], []
        for name, fm, variant in found:
            data = pd.read_csv(os.path.join(self.data_dir, name),
                               usecols=lambda c: c != 'FM')
            data['W'] = np.round(data['W'], W_DECIMALS)
            names = [n for n in MODEL_NAMES if f'M_{n}' in data.columns]
            print(f'[{variant}] FM={fm}: {len(data)} rows, P={sorted(data.P.unique())}, '
                  f'{data.Exp.nunique()} Exp replicas, models={names}')

            df_ms = self._mean_std_curves(data, names)
            df_I = self._compute_I(df_ms, names)

            w_anchor = 0.0
            if 'anchored' in data.columns:
                anch = data[data.anchored == 1]
                w_anchor = float(anch.W.max()) if not anch.empty else 0.0

            ms_out = os.path.join(self.data_dir, f'Projected_M_mean_std_{variant}_{fm}.csv')
            df_ms.to_csv(ms_out, index=False)
            print(f'  mean/std -> {ms_out}')

            if df_I is not None:
                I_out = os.path.join(self.data_dir, f'Projected_I_{variant}_{fm}.csv')
                df_I.to_csv(I_out, index=False)
                print(f'  I(omega) -> {I_out}')

            real_fm = None
            if df_real is not None:
                real_fm = df_real[df_real.FM.astype(str) == str(fm)]
                for model, (rmse, r2, rmse_mean, n) in self._test_error(data, real_fm, names).items():
                    err_rows.append({'variant': variant, 'FM': fm, 'model': model,
                                     'n_points': n, 'RMSE': rmse, 'R2': r2,
                                     'RMSE_mean': rmse_mean})
                    print(f'  test FM={fm:>3} {model:>3}  n={n:>6}  RMSE={rmse:.5f}  '
                          f'R2={r2:.5f}  RMSE(Exp-mean)={rmse_mean:.5f}')
            per_fm.append((fm, variant, df_ms, df_I, w_anchor, names, real_fm))

        # plot only the top-TOP_K models (by test error); all models if untestable
        top, overall = self._rank(err_rows) if err_rows else (None, None)
        for fm, variant, df_ms, df_I, w_anchor, names, real_fm in per_fm:
            plot_names = [n for n in top if n in names] if top else names
            self._plot(fm, variant, df_ms, df_I, w_anchor, plot_names, real_fm, overall)
        if top:
            self._plot_best(top[0], per_fm, overall)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='../Data/2026',
                        help="Directory holding Real_data_projected_{fm}_{variant}_v2.csv")
    parser.add_argument("--smooth_window", type=int, default=SMOOTH_WINDOW,
                        help="Savitzky-Golay window in omega-grid points (odd; larger = smoother)")
    parser.add_argument("--top_k", type=int, default=TOP_K,
                        help="Number of best-ranked models plotted in projected_I_*.png "
                             f"(default {TOP_K}; {len(MODEL_NAMES)} = all)")
    args = parser.parse_args()

    Test_Error(data_dir=args.data_dir, smooth_window=args.smooth_window, top_k=args.top_k).run()
