"""Post-process the reconstructed real-data M(omega) curves into I(omega).

Converted from test_error.ipynb and fitted to the current 2026 layout.

reconstruct_real_data.py writes one file per (FM, model variant):
    Data/2026/Real_data_projected_{fm}_no_exp_v2.csv     (rolled with *_exp_No)
    Data/2026/Real_data_projected_{fm}_with_exp_v2.csv    (rolled with *_exp_Yes)
each with columns  FM, P, Exp, W, anchored, M_<model> for every model that was
rolled (GB, RF, XGB, NN, GP -- whichever were trained).  This script
processes the files of one variant (--variant, default no_exp; 'both' runs
no_exp then with_exp) and, per FM:

  * averages each model's M over the Exp replicas -> Mean_M_* / Err_M_* per (P, W),
    Err = jackknife error over the Exp rows, sqrt(N-1) std (jackknife.py);
  * combines the momentum pairs in I_PAIRS (default only (1, 2), i.e.
    I = (4*M2 - M1) / 3; every P is still averaged, written and plotted as an
    M curve) into an Ioffe-time estimate
        I(w) = (p2^2 * M(p2, w) - p1^2 * M(p1, w)) / (p2^2 - p1^2)
    on the Exp-averaged mean curve, then averages the pair estimates at each w
    (negative values are kept, not clipped to 0).

  * smooths I(w) with a Savitzky-Golay filter (order-3 polynomial over a
    sliding window; --smooth_window points, default 60 -> 59 = 5.9 in omega).
    Savitzky-Golay keeps the single hump / peak position that a plain moving
    average (the old seasonal_decompose trend) rounded off, while removing the
    step artefacts the autoregressive roll leaves in the extrapolated tail.

  * writes, per (variant, FM):
        Data/2026/Projected_M_mean_err_{variant}_{fm}.csv
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
        Data/2026/test_error_summary_{variant}.csv    (per FM, per model)
        Data/2026/test_error_ranking_{variant}.csv    (overall ranking)
    and only the top three are plotted (M with the real data overlaid, and I) in
        Data/2026/Images/projected_I_{variant}_{fm}.png
    (--selectmodel NN RF plots exactly those models instead), plus every model in
        Data/2026/Images/projected_I_{variant}_{fm}_all.png
    The single best model's smoothed I(w) for every FM is also drawn on one axis:
        Data/2026/Images/best_model_I_{variant}_{model}.png
        Data/2026/best_model_I_{variant}_{model}.csv     (FM, W, smoothed I, jackknife Err)

Run from Helicity/Code:
    python test_error.py                       # no_exp (default)
    python test_error.py --variant with_exp
    python test_error.py --variant both
    python test_error.py --variant both --selectmodel NN RF
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

from jackknife import jk_error, jk_series_error

import warnings
warnings.filterwarnings("ignore")

pd.options.mode.chained_assignment = None

MODEL_NAMES = ['GB', 'RF', 'XGB', 'NN', 'GP', 'BNN', 'NNE', 'HNN', 'ET']
MODEL_TITLES = {'GB': 'Gradient boosted tree',
                'RF': 'Random forest',
                'XGB': 'Extreme gradient boosted tree',
                'NN': 'Neural network (MLP)',
                'GP': 'Gaussian process',
                'BNN': 'Bayesian neural network (HMC)',
                'NNE': 'NN bootstrap ensemble',
                'HNN': 'Heteroscedastic NN',
                'ET': 'Extra trees'}
VARIANT_TITLE = {'no_exp': 'without Exp feature', 'with_exp': 'with Exp feature'}
W_DECIMALS = 1
SMOOTH_WINDOW = 60       # Savitzky-Golay window in grid points (0.1 each)
SMOOTH_POLYORDER = 3
# momentum pairs (p1, p2) whose estimates are averaged into I(omega); M curves keep every P
I_PAIRS = [(1, 2)]
N_SEED = 4               # seed-window points per P in reconstruct_real_data.py (N_LAGS)
TOP_K = 3
# Default variant: the *_exp_No models (Exp is an arbitrary replica label, so
# feeding it as a feature only invites per-replica memorisation).  --variant
# with_exp evaluates the *_exp_Yes projections instead; 'both' runs each in turn.
VARIANT = 'no_exp'
VARIANTS = ['no_exp', 'with_exp']

_FILE_RE = re.compile(r'Real_data_projected_(.+)_(no_exp|with_exp)_v2\.csv$')


def best_model(data_dir, variant=VARIANT, model=None):
    """``model`` if given (``--model``), else the overall #1 in test_error_ranking_{variant}.csv.

    Same ranking run() uses to pick the model for best_model_I_{variant}_{model}.csv,
    so it needs only test_error.py (not train_error.py).
    """
    if model is not None:
        print(f'model: {model} (set by --model)')
        return model
    path = os.path.join(data_dir, f'test_error_ranking_{variant}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found -- run test_error.py first')
    best = pd.read_csv(path).sort_values('overall_rank').model.iloc[0]
    print(f'model: {best} (rank 1 overall in {path})')
    return best


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


def get_mean_err(data, grp_cols, agg_col):
    """Mean_{agg_col} / Err_{agg_col} (jackknife error over the Exp rows) grouped by ``grp_cols``."""
    g = data.groupby(grp_cols)[agg_col]
    df_mu = g.mean().reset_index().rename(columns={agg_col: f'Mean_{agg_col}'})
    df_err = g.agg(jk_series_error).reset_index().rename(columns={agg_col: f'Err_{agg_col}'})
    return df_mu.merge(df_err, on=grp_cols)


class Test_Error:
    def __init__(self, data_dir='../Data/2026', smooth_window=SMOOTH_WINDOW, top_k=TOP_K,
                 model=None, variant=VARIANT, select_models=None, ylim_I=None,
                 stat_from=None, sys_diff=False):
        self.ylim_I = ylim_I        # (lo, hi) fixed y-range of the I(omega) row; None = auto
        if variant not in VARIANTS:
            raise ValueError(f'variant {variant!r} not in {VARIANTS}')
        # models for projected_I_{variant}_{fm}.png; None = top_k by test error
        self.select_models = list(select_models) if select_models else None
        self.variant = variant      # 'no_exp' (*_exp_No models) or 'with_exp' (*_exp_Yes)
        self.model = model          # forced model for best_model_I_*; None = ranking #1
        self.data_dir = data_dir
        self.top_k = top_k
        self.smooth_window = smooth_window
        # stat_from='RF': every other model's statistical (jackknife) error is replaced
        # by that model's, combined in quadrature with its own model error (draws).
        # Error-carrying outputs then go to {data_dir}/stat_{model}/ instead.
        self.stat_from = stat_from
        # sys_diff (needs stat_from): add |model - stat_from model| as a systematic
        self.sys_diff = bool(sys_diff and stat_from)
        sub = f'stat_{stat_from}' + ('_sys' if self.sys_diff else '')
        self.out_dir = os.path.join(data_dir, sub) if stat_from else data_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.img_dir = os.path.join(self.out_dir, 'Images')

    def _discover(self):
        """[(filename, fm, variant), ...] for every projected file present."""
        out = []
        for name in sorted(os.listdir(self.data_dir)):
            m = _FILE_RE.match(name)
            if m and m.group(2) == self.variant:
                out.append((name, m.group(1), m.group(2)))
        return out

    def _mean_err_curves(self, data, names):
        """One frame indexed by (P, W) with Mean_M_* / Err_M_* per model.

        Err is the jackknife error over the Exp rows.  For a model with an
        M_{name}_draw column (reconstruct_real_data.py --propagate) the model
        error is added in quadrature: Err_model_M_* = plain std over Exp of
        (draw - deterministic), with no sqrt(N-1) because the draws are
        independent model samples, not jackknife samples.  The pure jackknife
        part is kept as Err_stat_M_*.
        """
        df = None
        for name in names:
            part = get_mean_err(data, ['P', 'W'], f'M_{name}')
            dcol = f'M_{name}_draw'
            if dcol in data.columns:
                diff = (data[dcol] - data[f'M_{name}']).groupby([data.P, data.W])
                sm = diff.std(ddof=0).rename(f'Err_model_M_{name}').reset_index()
                part = part.merge(sm, on=['P', 'W'])
                part[f'Err_stat_M_{name}'] = part[f'Err_M_{name}']
                part[f'Err_M_{name}'] = np.hypot(part[f'Err_stat_M_{name}'],
                                                 part[f'Err_model_M_{name}'])
            df = part if df is None else df.merge(part, on=['P', 'W'])
        src = self.stat_from
        if src and src in names:
            for name in names:
                if name == src:
                    continue
                stat = df[f'Err_stat_M_{src}'] if f'Err_stat_M_{src}' in df else df[f'Err_M_{src}']
                model = df.get(f'Err_model_M_{name}', 0.0)
                df[f'Err_stat_M_{name}'] = stat
                # model-difference systematic: |this model - stat_from model| per (P, W)
                sys_ = ((df[f'Mean_M_{name}'] - df[f'Mean_M_{src}']).abs() if self.sys_diff
                        else 0.0)
                df[f'Err_sys_M_{name}'] = sys_
                df[f'Err_M_{name}'] = np.sqrt(stat ** 2 + model ** 2 + sys_ ** 2)
                
        return df

    @staticmethod
    def _I_omega(piv, p1, p2):
        """('p1-p2', I(w) array) from one momentum pair of the pivot (index W, columns P)."""
        denom = p2 ** 2 - p1 ** 2
        col = f'{p1}-{p2}'
        vals = ((p2 ** 2 * piv[p2] - p1 ** 2 * piv[p1]) / denom).to_numpy()
        return col, vals

    def _I_rows(self, data, name, pairs, W, col=None):
        """Raw and smoothed I(w) on every Exp row (each a jackknife sample), with the
        same pair rule as _compute_I; each (n_Exp, n_W).  Negative values are kept.
        ``col`` overrides the M column (default M_{name})."""
        cube = data.set_index(['P', 'Exp', 'W'])[col or f'M_{name}'].unstack('W').reindex(columns=W)
        exps = np.sort(data.Exp.unique())
        M = {p: cube.loc[p].reindex(exps).to_numpy() for p in sorted({p for pr in pairs for p in pr})}
        E = np.stack([(j ** 2 * M[j] - i ** 2 * M[i]) / (j ** 2 - i ** 2) for i, j in pairs])
        raw = E.mean(axis=0)
        sm = np.apply_along_axis(get_smooth_curve, 1, raw, self.smooth_window)
        sm[:, W == 0.0] = 0.0
        return raw, sm

    def _compute_I(self, df_ms, names, data):
        """I(w) averaged over the ``I_PAIRS`` momentum pairs, + smoothed I(w),
        each with its jackknife error over the Exp rows (Err_I_*, Err_smooth_I*).

        Plain mean of the pair estimates at each W; negative values are kept (no clip at 0).
        """
        have = set(df_ms.P.unique())
        pairs = [(i, j) for i, j in I_PAIRS if i in have and j in have]
        if not pairs:
            print(f'  none of the I_PAIRS {I_PAIRS} present (P = {sorted(have)}); skipping I(omega)')
            return None

        rows = {}
        piv = None
        for name in names:
            piv = (df_ms.pivot_table(index='W', columns='P', values=f'Mean_M_{name}')
                   .sort_index())
            tmp = dict(self._I_omega(piv, i, j) for i, j in pairs)
            tmp = pd.DataFrame(tmp, index=piv.index)
            rows[f'I_{name}'] = tmp.mean(axis=1).to_numpy()
        rows['W'] = np.round(piv.index.to_numpy(), W_DECIMALS)

        df_I = pd.DataFrame(rows).sort_values('W').reset_index(drop=True)
        stat_I = {}                     # name -> (raw, smoothed) jackknife error only
        for name in names:
            sm = get_smooth_curve(df_I[f'I_{name}'], self.smooth_window)
            sm = np.where(df_I['W'].to_numpy() == 0.0, 0.0, sm)   # I(0) = 0 exactly
            df_I[f'smooth_I{name}'] = sm
            raw_rows, sm_rows = self._I_rows(data, name, pairs, df_I['W'].to_numpy())
            df_I[f'Err_I_{name}'] = jk_error(raw_rows, axis=0)
            df_I[f'Err_smooth_I{name}'] = jk_error(sm_rows, axis=0)
            stat_I[name] = (df_I[f'Err_I_{name}'].to_numpy(), df_I[f'Err_smooth_I{name}'].to_numpy())
            df_I[f'Err_model_I_{name}'] = 0.0
            df_I[f'Err_model_smooth_I{name}'] = 0.0
            if f'M_{name}_draw' in data.columns:
                # model error: plain std of (draw - deterministic) per replica, added
                # in quadrature to the jackknife error (see _mean_err_curves)
                raw_d, sm_d = self._I_rows(data, name, pairs, df_I['W'].to_numpy(),
                                           col=f'M_{name}_draw')
                e_raw = np.nanstd(raw_d - raw_rows, axis=0)
                e_sm = np.nanstd(sm_d - sm_rows, axis=0)
                df_I[f'Err_model_smooth_I{name}'] = e_sm
                df_I[f'Err_model_I_{name}'] = e_raw
                df_I[f'Err_I_{name}'] = np.hypot(df_I[f'Err_I_{name}'], e_raw)
                df_I[f'Err_smooth_I{name}'] = np.hypot(df_I[f'Err_smooth_I{name}'], e_sm)
        src = self.stat_from
        if src and src in names:
            for name in names:
                if name == src:
                    continue
                s_raw = ((df_I[f'I_{name}'] - df_I[f'I_{src}']).abs().to_numpy()
                         if self.sys_diff else 0.0)
                s_sm = ((df_I[f'smooth_I{name}'] - df_I[f'smooth_I{src}']).abs().to_numpy()
                        if self.sys_diff else 0.0)
                df_I[f'Err_sys_smooth_I{name}'] = s_sm
                stat_I[name] = stat_I[src]
                df_I[f'Err_I_{name}'] = np.sqrt(stat_I[src][0] ** 2
                                                + df_I[f'Err_model_I_{name}'] ** 2 + s_raw ** 2)
                df_I[f'Err_smooth_I{name}'] = np.sqrt(stat_I[src][1] ** 2
                                                      + df_I[f'Err_model_smooth_I{name}'] ** 2
                                                      + s_sm ** 2)
        for name in names:
            df_I[f'Err_stat_smooth_I{name}'] = stat_I[name][1]
            if f'Err_sys_smooth_I{name}' not in df_I:
                df_I[f'Err_sys_smooth_I{name}'] = 0.0
        return df_I[['W'] + [f'I_{n}' for n in names] + [f'smooth_I{n}' for n in names]
                    + [f'Err_I_{n}' for n in names] + [f'Err_smooth_I{n}' for n in names]
                    + [f'Err_stat_smooth_I{n}' for n in names]
                    + [f'Err_model_smooth_I{n}' for n in names]
                    + [f'Err_sys_smooth_I{n}' for n in names]]

    def _plot(self, fm, variant, df_ms, df_I, w_anchor, names, df_real=None, overall=None,
              *, suffix='', note=None):
        """M(w) (top row) and I(w) (bottom row), one column per model in ``names``.

        Projected M and smoothed I carry a +-1 sigma jackknife band; real lattice M
        is drawn as open markers with jackknife error bars.
        Each column title carries the model's overall test rank.
        Saved as projected_I_{variant}_{fm}{suffix}.png; ``note`` replaces the
        default 'top N models by test error' text in the suptitle.
        """
        ordered_ps = sorted(df_ms.P.unique())
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        real_ms = None
        if df_real is not None and not df_real.empty:
            real_ms = get_mean_err(df_real, ['P', 'W'], 'M')
        n_rows = 2 if df_I is not None else 1
        fig, axes = plt.subplots(n_rows, len(names), figsize=(8 * len(names), 9 * n_rows),
                                 dpi=330, squeeze=False, sharey='row')

        for col, name in enumerate(names):
            ax = axes[0, col]
            f_mu, f_sd = f'Mean_M_{name}', f'Err_M_{name}'
            for k, p in enumerate(ordered_ps):
                t = df_ms[df_ms.P == p].sort_values('W')
                c = colors[k % len(colors)]
                ax.plot(t.W, t[f_mu], '-', lw=1.5, color=c,
                        label=f'$p_z$={p}' if col == 0 else None)
                ax.fill_between(t.W, t[f_mu] - t[f_sd], t[f_mu] + t[f_sd],
                                color=c, alpha=0.25, lw=0)
                if real_ms is not None:
                    r = real_ms[real_ms.P == p].sort_values('W')
                    ax.errorbar(r.W, r.Mean_M, yerr=r.Err_M, fmt='o', mfc='none', ms=6,
                                capsize=3, color=c,
                                label=r'real data $\pm\sigma_{JK}$' if col == 0 and k == 0 else None)
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
                sm, e = df_I[f'smooth_I{name}'], df_I[f'Err_smooth_I{name}']
                em = df_I[f'Err_model_smooth_I{name}']
                esys = df_I[f'Err_sys_smooth_I{name}']
                es = df_I[f'Err_stat_smooth_I{name}']
                jk = (rf'\sigma_{{JK,{self.stat_from}}}' if self.stat_from and name != self.stat_from
                      else r'\sigma_{JK}')
                terms = [f'{jk}^2']
                if bool((em > 0).any()):
                    terms.append(r'\sigma_{model}^2')
                if bool((esys > 0).any()):
                    terms.append(rf'|I-I_{{{self.stat_from}}}|^2')
                axI.fill_between(df_I['W'], sm - e, sm + e, color=colors[3], alpha=0.3, lw=0,
                                 label=(rf'$\pm\sqrt{{{"+".join(terms)}}}$' if len(terms) > 1
                                        else rf'$\pm{jk}$'))
                if len(terms) > 1:
                    axI.fill_between(df_I['W'], sm - es, sm + es, color=colors[3], alpha=0.3,
                                     lw=0, label=rf'$\pm{jk}$ only')
                axI.plot(df_I['W'], sm, '-', lw=2, color=colors[3], label='smoothed')
                axI.axhline(0, ls=':', lw=1, color='k')
                if self.ylim_I:
                    axI.set_ylim(*self.ylim_I)
                if w_anchor and w_anchor > 0:
                    axI.axvline(w_anchor, ls='--', lw=1, color='0.5')
                axI.set_xlabel(r'$\omega$')
                if col == 0:
                    axI.set_ylabel(r'$\Delta\mathcal{I}_g(\omega)$')
                    axI.legend()

        anchor_note = (f'   (real data anchors $\\omega\\leq${w_anchor:.1f}, '
                       f'dashed line; beyond is extrapolation)' if w_anchor else '')
        if note is None:
            note = f'top {len(names)} models by test error' if overall is not None else ''
        top_note = f'{note}   ' if note else ''
        fig.suptitle(f'FM={fm}  [{VARIANT_TITLE.get(variant, variant)}]   {top_note}'
                     f'I($\\omega$) from $p_z$ pair(s) '
                     f'{", ".join(f"{i}-{j}" for i, j in I_PAIRS)}{anchor_note}')
        fig.tight_layout(rect=(0, 0, 1, 0.97))   # leave room for the suptitle
        os.makedirs(self.img_dir, exist_ok=True)
        out_png = os.path.join(self.img_dir, f'projected_I_{variant}_{fm}{suffix}.png')
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
            e = df_I[f'Err_smooth_I{best}'].to_numpy()
            c = colors[k % len(colors)]
            ax.fill_between(w, y - e, y + e, color=c, alpha=0.2, lw=0)
            ax.plot(w, y, styles[k % len(styles)], lw=2, color=c, label=f'FM={fm}')
            w_anchor = max(w_anchor, w_anc)
            curves.append(pd.DataFrame({'FM': fm, 'W': w, 'I': y, 'Err': e}))
        if curves:
            out_csv = os.path.join(self.out_dir, f'best_model_I_{self.variant}_{best}.csv')
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
        rank = int(o.overall_rank) if 'overall_rank' in o else None
        what = 'best mean test error' if rank in (None, 1) else f'rank {rank} by mean test error'
        ax.set_title(f'{MODEL_TITLES[best]} ({what}: RMSE={o.mean_RMSE:.4f}, '
                     f'$R^2$={o.mean_R2:.3f})\n'
                     f'smoothed I($\\omega$), {VARIANT_TITLE[self.variant]}', fontsize=10)
        ax.legend(frameon=False)
        # direct FM labels at the right end (omega ~18.5), where the curves overlap
        # less than at the peak; pushed apart vertically so they never collide
        if curves:
            lo, hi = ax.get_ylim()
            gap = 0.06 * (hi - lo)
            labs = []
            for k, cdf in enumerate(curves):
                w_c, y_c = cdf.W.to_numpy(), cdf.I.to_numpy()
                x_lab = min(18.5, w_c.max())
                labs.append([float(np.interp(x_lab, w_c, y_c)), x_lab, cdf.FM.iloc[0],
                             colors[k % len(colors)]])
            labs.sort(key=lambda t: t[0])
            for j in range(1, len(labs)):
                labs[j][0] = max(labs[j][0], labs[j - 1][0] + gap)
            for y_l, x_l, fm, c in labs:
                ax.text(x_l, y_l + 0.5 * gap, f'FM={fm}', ha='center', va='bottom',
                        fontsize=9, color=c)
        fig.tight_layout()
        out_png = os.path.join(self.img_dir, f'best_model_I_{self.variant}_{best}.png')
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
        out = os.path.join(self.data_dir, f'test_error_summary_{self.variant}.csv')
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
        out = os.path.join(self.data_dir, f'test_error_ranking_{self.variant}.csv')
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
        Writes Data/2026/train_test_error_{variant}.csv: one row per (FM, model)
        plus FM='all' rows with the means across FMs, ordered by overall rank.
        """
        path = os.path.join(self.data_dir, 'train_error_summary.csv')
        if not os.path.exists(path):
            print(f'{path} not found -- run train_error.py first; train/test csv skipped')
            return
        exp_id = 'Yes' if self.variant == 'with_exp' else 'No'
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
        out = os.path.join(self.data_dir, f'train_test_error_{self.variant}.csv')
        out_df.to_csv(out, index=False)
        print(f'\ntrain + test error -> {out}')
        print(out_df.to_string(index=False, float_format=lambda x: f'{x:.5g}'))

    def run(self):
        found = self._discover()
        if not found:
            print(f'no Real_data_projected_*_{self.variant}_v2.csv in {self.data_dir}')
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

            df_ms = self._mean_err_curves(data, names)
            df_I = self._compute_I(df_ms, names, data)

            w_anchor = 0.0
            if 'anchored' in data.columns:
                anch = data[data.anchored == 1]
                w_anchor = float(anch.W.max()) if not anch.empty else 0.0

            cum_sum = True
            if cum_sum:
                for col in df_ms.columns:
                    if col.startswith('Err_M_') and not col.endswith('RF'):
                        df_ms[col] = np.sqrt(np.cumsum(df_ms[col] ** 2))

            ms_out = os.path.join(self.out_dir, f'Projected_M_mean_err_{variant}_{fm}.csv')
            df_ms.to_csv(ms_out, index=False)
            print(f'  mean/jackknife err -> {ms_out}')

            if df_I is not None:
                if cum_sum:
                    for col in df_I.columns:
                        if col.startswith('Err_smooth_I') and not col.endswith('RF'):
                            df_I[col] = np.sqrt(np.cumsum(df_I[col] ** 2))

                I_out = os.path.join(self.out_dir, f'Projected_I_{variant}_{fm}.csv')
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

        # projected_I_{variant}_{fm}.png: --selectmodel models if given, else the
        # top-TOP_K by test error (all models if untestable).
        # projected_I_{variant}_{fm}_all.png: every model, ordered by overall rank.
        top, overall = self._rank(err_rows) if err_rows else (None, None)
        for fm, variant, df_ms, df_I, w_anchor, names, real_fm in per_fm:
            note = None
            if self.select_models:
                plot_names = [n for n in self.select_models if n in names]
                missing = [n for n in self.select_models if n not in names]
                if missing:
                    print(f'  FM={fm}: --selectmodel {missing} not in the projected file; skipped')
                if not plot_names:
                    print(f'  FM={fm}: no selected model available; projected_I plot skipped')
                note = f'selected models: {", ".join(plot_names)}'
            else:
                plot_names = [n for n in top if n in names] if top else names
            if plot_names:
                self._plot(fm, variant, df_ms, df_I, w_anchor, plot_names, real_fm, overall,
                           note=note)
            all_names = ([n for n in overall.index if n in names] if overall is not None
                         else names)
            self._plot(fm, variant, df_ms, df_I, w_anchor, all_names, real_fm, overall,
                       suffix='_all',
                       note='all models, by test rank' if overall is not None else 'all models')
        if top:
            chosen = self.model or top[0]
            if chosen not in overall.index:
                raise SystemExit(f'--model {chosen}: not in the projected files '
                                 f'(available: {list(overall.index)})')
            self._plot_best(chosen, per_fm, overall)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='../Data/2026',
                        help="Directory holding Real_data_projected_{fm}_{variant}_v2.csv")
    parser.add_argument("--smooth_window", type=int, default=SMOOTH_WINDOW,
                        help="Savitzky-Golay window in omega-grid points (odd; larger = smoother)")
    parser.add_argument("--top_k", type=int, default=TOP_K,
                        help="Number of best-ranked models plotted in projected_I_*.png "
                             f"(default {TOP_K}; {len(MODEL_NAMES)} = all)")
    parser.add_argument("--model", choices=MODEL_NAMES, default=None,
                        help="Model for best_model_I_*.{csv,png} (default: overall #1 by test error)")
    parser.add_argument("--variant", choices=VARIANTS + ['both'], default=VARIANT,
                        help="Which projections to evaluate: no_exp (*_exp_No models, default), "
                             "with_exp (*_exp_Yes models) or both")
    parser.add_argument("--selectmodel", nargs='+', default=None, metavar='MODEL',
                        help="Models drawn in projected_I_{variant}_{fm}.png instead of the "
                             f"top --top_k, e.g. --selectmodel NN RF (or NN,RF); one of "
                             f"{MODEL_NAMES}. projected_I_*_all.png always shows every model")
    parser.add_argument("--ylim_I", type=float, nargs=2, default=None, metavar=('LO', 'HI'),
                        help="fixed y-range of the I(omega) row in projected_I_*.png, e.g. --ylim_I -0.8 0.8")
    parser.add_argument("--stat_from", choices=MODEL_NAMES, default=None,
                        help="Use this model's jackknife (statistical) error for every model, "
                             "combined in quadrature with each model's own draw error; error "
                             "outputs/plots go to {data_dir}/stat_{model}/ (e.g. --stat_from RF)")
    parser.add_argument("--sys_diff", action='store_true',
                        help="With --stat_from M: add |model - M| (on M per (P, W) and on smoothed "
                             "I per omega) in quadrature as a model-difference systematic; "
                             "outputs go to {data_dir}/stat_{M}_sys/")
    args = parser.parse_args()

    select = None
    if args.selectmodel:
        select = [s.strip() for tok in args.selectmodel for s in tok.split(',') if s.strip()]
        bad = [s for s in select if s not in MODEL_NAMES]
        if bad:
            parser.error(f'--selectmodel: unknown model(s) {bad}; choose from {MODEL_NAMES}')
        select = list(dict.fromkeys(select))          # drop duplicates, keep order

    for variant in (VARIANTS if args.variant == 'both' else [args.variant]):
        print(f'\n===== variant: {variant} ({VARIANT_TITLE[variant]}) =====')
        Test_Error(data_dir=args.data_dir, smooth_window=args.smooth_window, top_k=args.top_k,
                   model=args.model, variant=variant, select_models=select,
                   ylim_I=args.ylim_I, stat_from=args.stat_from,
                   sys_diff=args.sys_diff).run()
