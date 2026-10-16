"""QAIC model selection among candidate forms for the real M(omega, P).

Every model is fit, per lattice ensemble FM, to the same cells: the real
per-(P, omega) mean M over the Exp replicas, P = 1..5, omega > 0 (the W=0
anchor rows are excluded), with weight 1 / err^2, err = the jackknife error of the
cell mean over Exp (jackknife.py: the Exp rows are jackknife samples, err = sqrt(n-1) std)
-- the likelihood used by synthetic_data_generator.generate_I_omega.

    M1  A w e^{-s w} P^{-g} e^{c z2}          (the model in generate_I_omega)
    M2  A w e^{-s w}                          no P dependence
    M3  A w e^{-s w} P^{-g}                   no z2 term
    M4  A w e^{-s w} e^{c z2}                 no P power
    M5  A w e^{-s w} P^{-g} (1 + c z2)        linear O(z^2) correction
    M6  A w^a e^{-s w} P^{-g} e^{c z2}        free omega power (M1 at a = 1)
    M7  A w / (1 + (w/w0)^2) P^{-g} e^{c z2}  rational omega shape
    M8  A w e^{-(w/w0)^2} P^{-g} e^{c z2}     Gaussian omega fall-off
    with z2 = min((w / P)^2, z2_max), z2_max = the largest (w/P)^2 in the cells.

With Gaussian errors of known sigma, -2 ln L = chi^2 + const, so
    AIC  = chi^2 + 2k
    AICc = AIC + 2k(k+1) / (N - k - 1)     (N = cells per FM, only ~15-25)
    BIC  = chi^2 + k ln N
Each fit is multi-started within wide bounds and the lowest chi^2 is kept.

The selection criterion is QAICc.  With jackknife cell errors chi^2/dof is O(1),
but neighbouring omega cells are correlated (same configurations) and that
correlation is not in the diagonal chi^2.  QAIC guards against the resulting
overdispersion by dividing chi^2 by c_hat = chi^2/dof of the most flexible model
(M6) in that FM, counting c_hat as one extra parameter (k' = k + 1):
    QAIC  = chi^2 / c_hat + 2k'
    QAICc = QAIC + 2k'(k'+1) / (N - k' - 1)
AIC/AICc/BIC and the unweighted per-P RMSE (mean over P = model error per FM)
are still written for reference.

--n_boot B resamples the Exp rows with replacement (jointly across P within an
FM; c_hat kept from the full sample), refits every model on each resample and
counts how often each model is best by QAICc.  The rows are jackknife samples,
so the resampled mean's deviation is scaled by (n - 1) (= resampling the
jackknife pseudo-values) to give the cell means their true spread.

Writes (Data/2026/):
    model_selection_fits.csv      per (FM, model): k, N, chi2, chi2_dof, c_hat, QAIC(c),
                                  Delta_QAICc, w_QAICc, AIC(c)/BIC, mean (n)RMSE, parameters
    model_selection_summary.csv   per model, sorted by summed QAICc over FMs
    model_selection_rmse_per_P.csv  per (FM, model, P): RMSE, nRMSE
    model_selection_bootstrap.csv per (FM, model): share of resamples won by QAICc
    Images/model_selection_FM_{fm}.png   data vs the best models by QAICc, one panel per P

best_model() reads these back and returns the selected form (highest bootstrap
QAICc win fraction, ties by summed QAICc); `--curves` with no names plots it.

Run from Helicity/Code:  python model_selection.py [--n_boot 200]
                         python model_selection.py --curves    -> M_curves_<best>.csv
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from jackknife import jk_error

warnings.filterwarnings("ignore")

DATA_DIR = '../Data/2026/'
IMG_DIR = os.path.join(DATA_DIR, 'Images')
PLOT_DPI = 330
P_SET = [1, 2, 3, 4, 5]
N_STARTS = 30


def load_real_data():
    """Read All_p_w_m.csv, building it from rpitd/*.txt first if it is missing."""
    path = os.path.join(DATA_DIR, 'All_p_w_m.csv')
    if not os.path.exists(path):
        # imported here: synthetic_data_generator imports this module; rc_context
        # undoes the rcParams it sets on import
        with matplotlib.rc_context():
            from synthetic_data_generator import create_combined_data
        create_combined_data()
    return pd.read_csv(path)


def _z2(w, P, z2_max):
    return np.minimum((w / P) ** 2, z2_max)


# name -> (formula label, parameter names, lower bounds, upper bounds, f(w, P, theta, z2_max))
MODELS = {
    'M1': (r'$A\omega e^{-s\omega} P^{-\gamma} e^{c z^2}$', ['A', 's', 'gamma', 'c'],
           [0, -2, -2, -10], [50, 5, 10, 10],
           lambda w, P, t, zm: t[0] * w * np.exp(-t[1] * w) * P ** (-t[2]) * np.exp(t[3] * _z2(w, P, zm))),
    'M2': (r'$A\omega e^{-s\omega}$', ['A', 's'],
           [0, -2], [50, 5],
           lambda w, P, t, zm: t[0] * w * np.exp(-t[1] * w)),
    'M3': (r'$A\omega e^{-s\omega} P^{-\gamma}$', ['A', 's', 'gamma'],
           [0, -2, -2], [50, 5, 10],
           lambda w, P, t, zm: t[0] * w * np.exp(-t[1] * w) * P ** (-t[2])),
    'M4': (r'$A\omega e^{-s\omega} e^{c z^2}$', ['A', 's', 'c'],
           [0, -2, -10], [50, 5, 10],
           lambda w, P, t, zm: t[0] * w * np.exp(-t[1] * w) * np.exp(t[2] * _z2(w, P, zm))),
    'M5': (r'$A\omega e^{-s\omega} P^{-\gamma} (1 + c z^2)$', ['A', 's', 'gamma', 'c'],
           [0, -2, -2, -10], [50, 5, 10, 10],
           lambda w, P, t, zm: t[0] * w * np.exp(-t[1] * w) * P ** (-t[2]) * (1 + t[3] * _z2(w, P, zm))),
    'M6': (r'$A\omega^{a} e^{-s\omega} P^{-\gamma} e^{c z^2}$', ['A', 'a', 's', 'gamma', 'c'],
           [0, 0.1, -2, -2, -10], [50, 4, 5, 10, 10],
           lambda w, P, t, zm: t[0] * w ** t[1] * np.exp(-t[2] * w) * P ** (-t[3]) * np.exp(t[4] * _z2(w, P, zm))),
    'M7': (r'$A\omega / (1 + (\omega/\omega_0)^2) P^{-\gamma} e^{c z^2}$', ['A', 'w0', 'gamma', 'c'],
           [0, 0.05, -2, -10], [50, 100, 10, 10],
           lambda w, P, t, zm: t[0] * w / (1 + (w / t[1]) ** 2) * P ** (-t[2]) * np.exp(t[3] * _z2(w, P, zm))),
    'M8': (r'$A\omega e^{-(\omega/\omega_0)^2} P^{-\gamma} e^{c z^2}$', ['A', 'w0', 'gamma', 'c'],
           [0, 0.05, -2, -10], [50, 100, 10, 10],
           lambda w, P, t, zm: t[0] * w * np.exp(-(w / t[1]) ** 2) * P ** (-t[2]) * np.exp(t[3] * _z2(w, P, zm))),
}


def load_cells(data, fm):
    """Real cells of one FM: P, W, mean, err (jackknife), n, weight (= 1 / err^2), and the
    Exp x cell matrix of M used to rebuild cell means under bootstrap."""
    d = data[(data.FM == fm) & (data.W > 0) & data.P.isin(P_SET)]
    piv = d.pivot_table(index='Exp', columns=['P', 'W'], values='M')
    cells = piv.columns.to_frame(index=False)
    cells['mean'] = piv.mean().to_numpy()
    cells['err'] = jk_error(piv.to_numpy(), axis=0)
    cells['n'] = piv.count().to_numpy()
    cells['weight'] = 1.0 / cells['err'] ** 2
    return cells, piv.to_numpy()


def fit_model(name, w, P, y, sw, z2_max, rng, x0=None, n_starts=N_STARTS):
    """Weighted least squares, multi-started; returns (theta, chi2)."""
    _, _, lo, hi, f = MODELS[name]
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)

    def resid(t):
        r = sw * (f(w, P, t, z2_max) - y)
        return np.where(np.isfinite(r), r, 1e12)

    starts = [] if x0 is None else [np.clip(x0, lo, hi)]
    # starts drawn uniformly inside a "sensible" sub-box of the bounds
    s_lo = np.maximum(lo, -2.0)
    s_hi = np.minimum(hi, 5.0)
    starts += [rng.uniform(s_lo, s_hi) for _ in range(n_starts)]

    best = None
    for x in starts:
        try:
            r = least_squares(resid, x, bounds=(lo, hi), method='trf', max_nfev=4000)
        except ValueError:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best.x, 2.0 * best.cost


def scores(chi2, k, n):
    aic = chi2 + 2 * k
    aicc = aic + 2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else np.inf
    bic = chi2 + k * np.log(n)
    return aic, aicc, bic


def qaic_scores(chi2, k, n, c_hat):
    kq = k + 1                                  # + c_hat
    qaic = chi2 / c_hat + 2 * kq
    qaicc = qaic + 2 * kq * (kq + 1) / (n - kq - 1) if n - kq - 1 > 0 else np.inf
    return qaic, qaicc


def add_deltas(df, by=None):
    for s in ('QAICc', 'AIC', 'AICc', 'BIC'):
        ref = df.groupby(by)[s].transform('min') if by else df[s].min()
        df[f'Delta_{s}'] = df[s] - ref
    for s in ('QAICc', 'AICc'):
        rel = np.exp(-0.5 * df[f'Delta_{s}'])
        df[f'w_{s}'] = rel / (rel.groupby(df[by]).transform('sum') if by else rel.sum())
    return df


def plot_fm(fm, cells, fits, z2_max, top):
    ps = sorted(cells.P.unique())
    fig, axes = plt.subplots(1, len(ps), figsize=(4.2 * len(ps), 4.2), dpi=PLOT_DPI, squeeze=False)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for ax, p in zip(axes.flat, ps):
        c = cells[cells.P == p]
        ax.errorbar(c.W, c['mean'], yerr=c['err'], fmt='o', color='k', ms=5, capsize=3, label='real')
        wg = np.linspace(1e-3, c.W.max() * 1.1, 200)
        for k, name in enumerate(top):
            theta = fits[name]
            ax.plot(wg, MODELS[name][4](wg, float(p), theta, z2_max), '-', lw=1.6,
                    color=colors[k % len(colors)], label=name)
        ax.set_title(fr'$p_z$={p}')
        ax.set_xlabel(r'$\omega$')
    axes[0, 0].set_ylabel(r'$\Delta\mathfrak{M}(\omega)$')
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(f'FM={fm}: real cell means vs best models by QAICc')
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, f'model_selection_FM_{fm}.png'), dpi=PLOT_DPI)
    plt.close(fig)


def plot_compare(names):
    """Real cell means vs the saved fits of ``names`` (model_selection_fits.csv),
    every FM x P; under each M row, the residual (model - real) in units of the
    cell's jackknife error.  Writes Images/model_selection_compare_{names}.png."""
    data = load_real_data()
    fits = pd.read_csv(os.path.join(DATA_DIR, 'model_selection_fits.csv'))
    fms = sorted(data.FM.unique())
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e']

    fig, axes = plt.subplots(2 * len(fms), len(P_SET), figsize=(4.2 * len(P_SET), 3.4 * 2 * len(fms)),
                             dpi=PLOT_DPI, squeeze=False,
                             gridspec_kw={'height_ratios': [3, 1.3] * len(fms)})
    for r, fm in enumerate(fms):
        cells, _ = load_cells(data, fm)
        z2_max = float(np.max((cells.W / cells.P) ** 2))
        for col, p in enumerate(P_SET):
            ax, axr = axes[2 * r, col], axes[2 * r + 1, col]
            c = cells[cells.P == p]
            wg = np.linspace(1e-3, c.W.max() * 1.1, 200)
            ax.errorbar(c.W, c['mean'], yerr=c['err'], fmt='o', color='k', ms=5, capsize=3,
                        label='real', zorder=3)
            for k, name in enumerate(names):
                row = fits[(fits.FM == fm) & (fits.model == name)].iloc[0]
                theta = [row[f'p_{pn}'] for pn in MODELS[name][1]]
                f = MODELS[name][4]
                ls = '-' if k == 0 else '--'
                ax.plot(wg, f(wg, float(p), theta, z2_max), ls, lw=1.8, color=colors[k],
                        label=f'{name} ($\\chi^2$/dof={row.chi2_dof:.2g})')
                pull = (f(c.W.to_numpy(), float(p), theta, z2_max) - c['mean']) / c['err']
                axr.plot(c.W, pull, 'o' + ls, color=colors[k], ms=4, lw=1)
            axr.axhline(0, color='0.5', lw=0.8)
            ax.set_title(f'FM={fm}  $p_z$={p}')
            if col == 0:
                ax.set_ylabel(r'$\Delta\mathfrak{M}(\omega)$')
                axr.set_ylabel('(model-real)/err')
                ax.legend(fontsize=8)
            axr.set_xlabel(r'$\omega$')
    fig.suptitle(' vs '.join(f'{n}: {MODELS[n][0]}' for n in names), fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = os.path.join(IMG_DIR, f'model_selection_compare_{"_".join(names)}.png')
    fig.savefig(out, dpi=PLOT_DPI)
    plt.close(fig)
    print(f'plot -> {out}')


def plot_curves(name, w_grid=np.round(np.arange(0.0, 20.05, 0.1), 1)):
    """M(w, P) of one saved fit (model_selection_fits.csv) on ``w_grid`` for every
    FM and P in P_SET, one panel per FM, real cell means overlaid.  z2 keeps the
    fit's cap z2_max (largest (w/P)^2 of that FM's cells).
    Writes M_curves_{name}.csv (FM, P, W, M) and Images/M_curves_{name}.png."""
    data = load_real_data()
    fits = pd.read_csv(os.path.join(DATA_DIR, 'model_selection_fits.csv'))
    fms = sorted(data.FM.unique())
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    f = MODELS[name][4]

    fig, axes = plt.subplots(1, len(fms), figsize=(7 * len(fms), 6), dpi=PLOT_DPI, squeeze=False)
    parts = []
    for ax, fm in zip(axes.flat, fms):
        cells, _ = load_cells(data, fm)
        z2_max = float(np.max((cells.W / cells.P) ** 2))
        row = fits[(fits.FM == fm) & (fits.model == name)].iloc[0]
        theta = [row[f'p_{pn}'] for pn in MODELS[name][1]]
        for k, p in enumerate(P_SET):
            m = f(w_grid, float(p), theta, z2_max)
            parts.append(pd.DataFrame({'FM': fm, 'P': p, 'W': w_grid, 'M': m}))
            col = colors[k % len(colors)]
            ax.plot(w_grid, m, '-', lw=1.8, color=col, label=fr'$p_z$={p}')
            c = cells[cells.P == p]
            ax.errorbar(c.W, c['mean'], yerr=c['err'], fmt='o', color=col, mfc='none',
                        ms=6, capsize=2)
        ax.axhline(0, color='0.6', lw=0.6)
        ax.set_title(f'FM={fm}   ' + ', '.join(f'{pn}={v:.3g}' for pn, v in
                                               zip(MODELS[name][1], theta)), fontsize=11)
        ax.set_xlabel(r'$\omega$')
    axes[0, 0].set_ylabel(r'$\Delta\mathfrak{M}(\omega, p_z)$')
    axes[0, 0].legend(title='lines: fit, circles: real', fontsize=10)
    fig.suptitle(f'{name}: {MODELS[name][0]}', fontsize=14)
    fig.tight_layout()
    out = os.path.join(IMG_DIR, f'M_curves_{name}.png')
    fig.savefig(out, dpi=PLOT_DPI)
    plt.close(fig)
    pd.concat(parts, ignore_index=True).to_csv(os.path.join(DATA_DIR, f'M_curves_{name}.csv'), index=False)
    print(f'plot -> {out}')


def best_model():
    """The selected M(omega, P) form: highest mean bootstrap QAICc win fraction over FMs
    (model_selection_bootstrap.csv), ties broken by summed QAICc (model_selection_summary.csv);
    summed QAICc alone if the bootstrap was skipped (--n_boot 0)."""
    summ = os.path.join(DATA_DIR, 'model_selection_summary.csv')
    if not os.path.exists(summ):
        raise FileNotFoundError(f'{summ} not found -- run  python model_selection.py  first')
    rank = pd.read_csv(summ).set_index('model')[['QAICc']]
    boot = os.path.join(DATA_DIR, 'model_selection_bootstrap.csv')
    if os.path.exists(boot):
        rank['win'] = pd.read_csv(boot).groupby('model').win_frac_QAICc.mean()
        best = rank.sort_values(['win', 'QAICc'], ascending=[False, True]).index[0]
        print(f'best model: {best} (bootstrap QAICc win fraction {rank.win[best]:.3f}, {boot})')
    else:
        best = rank.QAICc.idxmin()
        print(f'best model: {best} (lowest summed QAICc, {summ}; no bootstrap file)')
    return best


def main(n_boot, seed):
    os.makedirs(IMG_DIR, exist_ok=True)
    data = load_real_data()
    rng = np.random.default_rng(seed)

    rows, boot_rows, rmse_rows = [], [], []
    for fm in sorted(data.FM.unique()):
        cells, Mmat = load_cells(data, fm)
        w, P = cells.W.to_numpy(float), cells.P.to_numpy(float)
        y, sw = cells['mean'].to_numpy(), np.sqrt(cells['weight'].to_numpy())
        z2_max = float(np.max((w / P) ** 2))
        N = len(cells)

        fits = {}
        for name, (label, pnames, *_rest) in MODELS.items():
            theta, chi2 = fit_model(name, w, P, y, sw, z2_max, rng)
            fits[name] = theta
            k = len(pnames)
            aic, aicc, bic = scores(chi2, k, N)

            # unweighted error of the fitted curve vs the real cell means, per P
            pred = MODELS[name][4](w, P, theta, z2_max)
            per_p = []
            for p in P_SET:
                m = P == p
                rmse = float(np.sqrt(np.mean((pred[m] - y[m]) ** 2)))
                nrmse = rmse / float(np.sqrt(np.mean(y[m] ** 2)))
                per_p.append((rmse, nrmse))
                rmse_rows.append({'FM': fm, 'model': name, 'P': p, 'n': int(m.sum()),
                                  'RMSE': rmse, 'nRMSE': nrmse})

            rows.append({'FM': fm, 'model': name, 'formula': label, 'k': k, 'N': N,
                         'chi2': chi2, 'chi2_dof': chi2 / max(N - k, 1),
                         'AIC': aic, 'AICc': aicc, 'BIC': bic,
                         'mean_RMSE': float(np.mean([r for r, _ in per_p])),
                         'mean_nRMSE': float(np.mean([n for _, n in per_p])),
                         **{f'p_{pn}': v for pn, v in zip(pnames, theta)}})
            print(f'FM={fm:>2} {name}: chi2={chi2:12.1f}  AICc={aicc:12.1f}  BIC={bic:12.1f}')

        # QAIC: chi2 rescaled by the overdispersion c_hat = chi2/dof of the most
        # flexible model in this FM, so a misfit common to all models no longer
        # counts as evidence
        fm_rows = [r for r in rows if r['FM'] == fm]
        c_hat = max(fm_rows, key=lambda r: r['k'])['chi2_dof']
        for r in fm_rows:
            r['c_hat'] = c_hat
            r['QAIC'], r['QAICc'] = qaic_scores(r['chi2'], r['k'], N, c_hat)

        top = sorted(fm_rows, key=lambda r: r['QAICc'])[:3]
        top = [r['model'] for r in top]
        if 'M1' not in top:
            top.append('M1')
        plot_fm(fm, cells, fits, z2_max, top)

        if n_boot > 0:
            wins = dict.fromkeys(MODELS, 0)
            n_exp = Mmat.shape[0]
            for b in range(n_boot):
                idx = rng.integers(0, n_exp, n_exp)
                # rows are jackknife samples: scale the deviation by (n-1) (pseudo-values);
                # sigma and c_hat kept from the full sample
                yb = y + (n_exp - 1) * (np.nanmean(Mmat[idx], axis=0) - y)
                sc = {}
                for name, (_, pnames, *_rest) in MODELS.items():
                    _, chi2 = fit_model(name, w, P, yb, sw, z2_max, rng, x0=fits[name], n_starts=3)
                    sc[name] = qaic_scores(chi2, len(pnames), N, c_hat)[1]
                wins[min(sc, key=sc.get)] += 1
            for name in MODELS:
                boot_rows.append({'FM': fm, 'model': name, 'n_boot': n_boot,
                                  'win_frac_QAICc': wins[name] / n_boot})
            print(f'FM={fm} bootstrap wins (QAICc): ' +
                  ', '.join(f'{m}={v}' for m, v in wins.items() if v))

    df = add_deltas(pd.DataFrame(rows), by='FM')
    df.to_csv(os.path.join(DATA_DIR, 'model_selection_fits.csv'), index=False)
    pd.DataFrame(rmse_rows).to_csv(os.path.join(DATA_DIR, 'model_selection_rmse_per_P.csv'), index=False)

    summ = (df.groupby('model')
            .agg(k=('k', 'first'), N=('N', 'sum'), chi2=('chi2', 'sum'), QAICc=('QAICc', 'sum'),
                 AIC=('AIC', 'sum'), AICc=('AICc', 'sum'), BIC=('BIC', 'sum'),
                 mean_RMSE=('mean_RMSE', 'mean'), mean_nRMSE=('mean_nRMSE', 'mean'))
            .reset_index())
    summ = add_deltas(summ).sort_values('QAICc').reset_index(drop=True)
    summ.to_csv(os.path.join(DATA_DIR, 'model_selection_summary.csv'), index=False)

    print('\nc_hat (overdispersion) per FM:',
          df.groupby('FM').c_hat.first().round(0).to_dict())
    for col in ('Delta_QAICc', 'w_QAICc', 'mean_RMSE'):
        print(f'\n{col} per FM:')
        print(df.pivot_table(index='model', columns='FM', values=col).round(4).to_string())
    print('\nsummed over FMs (sorted by QAICc):')
    print(summ[['model', 'k', 'Delta_QAICc', 'w_QAICc', 'Delta_AICc', 'mean_RMSE']]
          .round(4).to_string(index=False))

    if boot_rows:
        dfb = pd.DataFrame(boot_rows)
        dfb.to_csv(os.path.join(DATA_DIR, 'model_selection_bootstrap.csv'), index=False)
        print('\nbootstrap: share of resamples each model wins by QAICc')
        print(dfb.pivot_table(index='model', columns='FM', values='win_frac_QAICc').round(3).to_string())

def estimate_M_values(Mk, omega, n_p=5, lr=0.1, epoch=100000, tol=1e-5):
    """Take M at P=k=3 as the best match and estimate M at every other P so all
    pairs give the same I under M_P = I + Y / P^2.
    M1, M2 (i=1, j=2) are found so the (i, k) and (j, k) pairs give the same I,
        A Mk + (1 - A) Mi = B Mk + (1 - B) Mj,  A = k^2/(k^2-i^2), B = k^2/(k^2-j^2),
    i.e. Mk = (1 - C) Mi + C Mj. I > 0 needs Mi < (k^2/i^2) Mk, so Mi and Mj start
    at half of (k^2/P^2) Mk (the I = 0 point) and gradient descent on
    r^2, r = Mk - (1 - C) Mi - C Mj, adjusts them; stops after `epoch` steps or once
    loss_M changes by less than `tol`. Stable only for lr < 1 / ((1 - C)^2 + C^2) (~0.69).
    Then I = (k^2 Mk - i^2 Mi) / (k^2 - i^2) and, for P = k+1..n_p,
    M_P = (I + (A_P - 1) Mi) / A_P with A_P = P^2 / (P^2 - i^2).
    Mk and omega are arrays on the same omega grid.
    Returns a dict with W, M1..M{n_p} and I, and a DataFrame of loss_M per epoch
    (RMSE of the Mk implied by Mi, Mj against Mk)."""

    if lr <= 0 or epoch < 1:
        raise ValueError(f'need lr > 0 and epoch >= 1, got lr={lr}, epoch={epoch}')
    
    i,j,k=1,2,3

    Mk = np.array(Mk, dtype=float)
    Mi = np.array((k**2/i**2)*Mk*0.5, dtype=float)
    Mj = np.array((k**2/j**2)*Mk*0.5, dtype=float)

    A = j**2 / (j**2 - i**2)
    B = k**2 / (k**2 - i**2)
    C = A / B
    D = i**2 / k**2

    loss = []
    prev = np.inf
    for ep in range(1, epoch + 1):
        resid = Mk - ((1-C) * Mi + C * Mj)
        # dL/dMi = -2(1-C)r, dL/dMj = -2Cr
        Mi = Mi + 2 * lr * resid * (1-C)
        Mj = Mj + 2 * lr * resid * C

        Est_Mk = (1-C) * Mi + C * Mj
        loss_M = np.sqrt(np.mean((Mk - Est_Mk) ** 2))
        loss.append({'epoch': ep, 'loss_M': loss_M})
        
        if abs(prev - loss_M) < tol:
            break
        prev = loss_M

    I = (Mk - D*Mi) / (1-D)

    m_dict = {'W':omega, 'M1':Mi, 'M2':Mj, 'M3':Mk}

    for p in range(k+1, n_p+1):
        A = p**2 / (p**2 - i**2)
        m_dict[f'M{p}'] = (I + (A-1)*Mi) / A

    m_dict['I'] = I

    return m_dict, pd.DataFrame(loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_boot', type=int, default=200,
                        help='Exp-bootstrap resamples per FM (0 = off)')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--compare', nargs='+', choices=list(MODELS), default=None,
                        help='Only plot these models from the saved model_selection_fits.csv '
                             '(no refit), e.g. --compare M1 M8')
    parser.add_argument('--curves', nargs='*', choices=list(MODELS), default=None,
                        help='Only plot M(w, P) on w = 0..20 (step 0.1) from the saved fits, one '
                             'figure per model.  Bare --curves = the best model (best_model()); '
                             'names plot those instead, e.g. --curves M1 M8')
    args = parser.parse_args()
    if args.curves is not None:
        for name in args.curves or [best_model()]:
            plot_curves(name)
    elif args.compare:
        plot_compare(args.compare)
    else:
        main(args.n_boot, args.seed)
