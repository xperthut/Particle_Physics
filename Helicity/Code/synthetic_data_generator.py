"""Generate synthetic M(omega, P) replicas per lattice ensemble (``FM`` = 9 / 12 / 15).

Stages run by ``__main__``:

1. ``create_combined_data`` (only if ``All_p_w_m.csv`` does not exist yet) -- concatenate the raw per-momentum RpITD files
   into ``Data/2026/All_p_w_m.csv`` (with a single omega=0 anchor per (FM, P)),
   keeping only the ``N_OMEGA_PER_FM[fm]`` smallest omega values per (FM, P).

2. ``generate_m_omega`` -- per FM, take M at P=3 from the selected model's
   curve (``M_curves_{MK_MODEL}.csv``, written by ``model_selection.py --curves``)
   as the best match and estimate every other P with
   ``model_selection.estimate_M_values``, so every momentum pair gives the same
   I(omega) under M_P = I + Y / P^2.  Writes ``new_M.csv`` (omega = 0..20),
   ``I_omega_estimated.csv``, ``loss_val_FM_{fm}.csv`` and
   ``M_real_vs_est_FM_{fm}.csv``.

3. ``generate_m_omega_from_distribution`` -- draw synthetic Exp replicas of M
   around ``new_M.csv`` with a linear std(omega) model, written to
   ``generated_Synthetic_exp_data.csv`` (read by prepare_train_test_data.py).

Run with ``Helicity/Code`` as the working directory:  ``python synthetic_data_generator.py``
(run ``python model_selection.py --curves M3`` first if ``M_curves_M3.csv`` is missing).
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import savgol_filter

from model_selection import estimate_M_values

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

# model_selection fit whose P=3 curve is taken as the best match (Mk)
MK_MODEL = 'M3'
MK_P = 3
N_P = 5

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

    # a single W=0, M=0 anchor row per (FM, P, Exp): omega=0 is Z=0
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

    data.to_csv(os.path.join(DATA_DIR, 'All_p_w_m.csv'), index=False)
    return data


def real_mean_M(data, fm):
    """Mean/std real M per (P, W) for one ensemble."""
    df_g = (
        data[data.FM == fm]
        .groupby(['P', 'W'])['M'].agg(Mean_M='mean', Std_M='std')
        .reset_index()
    )
    df_g['Std_M'] = df_g['Std_M'].fillna(0.0)
    return df_g


def load_mk(fm):
    """``(W, Mk)`` of the ``MK_MODEL`` curve at P = ``MK_P`` for ensemble ``fm``."""
    fp = os.path.join(DATA_DIR, f'M_curves_{MK_MODEL}.csv')
    if not os.path.exists(fp):
        raise FileNotFoundError(f'{fp} not found -- run  python model_selection.py --curves {MK_MODEL}')
    df = pd.read_csv(fp)
    g = df[(df.FM == int(fm)) & (df.P == MK_P)].sort_values('W')
    if g.empty:
        raise ValueError(f'{fp} has no P={MK_P} rows for FM={fm}.')
    return g['W'].to_numpy(dtype=float), g['M'].to_numpy(dtype=float)


def plot_I_omega(df_I):
    plt.figure(dpi=PLOT_DPI)
    for fm in sorted(df_I.FM.unique()):
        plt.plot(df_I[df_I.FM == fm].W, df_I[df_I.FM == fm].I, '-', label=f'FM={fm / 100:.2f}')
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\Delta \mathcal{I}(\omega)$')
    plt.legend()
    plt.savefig(os.path.join(IMG_DIR, 'I(w)_estimated.png'), format='png', dpi=PLOT_DPI)
    plt.close()


def plot_real_vs_synthetic(df_g, est_full, fm, full_range=False):
    """One panel per P: real mean M (+/- std) points vs the synthetic M(omega) curve.
    ``full_range`` -- if True show omega up to 20, else zoom to the real data range."""
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

    fig.suptitle(f'FM={fm}   M at P={MK_P} from {MK_MODEL}')
    fig.tight_layout()
    suffix = '_full' if full_range else ''
    fig.savefig(os.path.join(IMG_DIR, f'M_real_vs_synthetic_FM_{fm}{suffix}.png'),
                format='png', dpi=PLOT_DPI)
    plt.close(fig)


def generate_m_omega():
    """Per FM: Mk = the ``MK_MODEL`` curve at P = ``MK_P`` (omega = 0..20), then
    M1..M{N_P} and I from ``estimate_M_values``.

    Writes ``new_M.csv`` (FM, P, W, Est_M), ``I_omega_estimated.csv`` (FM, W, I),
    ``loss_val_FM_{fm}.csv`` (epoch, loss_M) and ``M_real_vs_est_FM_{fm}.csv``
    (real cell means vs the estimate interpolated onto the real omega).
    """
    data = pd.read_csv(os.path.join(DATA_DIR, 'All_p_w_m.csv'))

    all_new_M, all_I = [], []
    for fm in sorted(data.FM.unique()):
        df_g = real_mean_M(data, fm)
        W, Mk = load_mk(fm)

        m_dict, df_loss = estimate_M_values(Mk, W, n_p=N_P)
        df_loss.to_csv(os.path.join(DATA_DIR, f'loss_val_FM_{fm}.csv'), index=False)

        est_full = pd.concat([pd.DataFrame({'P': p, 'W': W, 'Est_M': m_dict[f'M{p}']})
                              for p in range(1, N_P + 1)], ignore_index=True)
        new_M = est_full.copy()
        new_M['FM'] = fm
        all_new_M.append(new_M)
        all_I.append(pd.DataFrame({'FM': fm, 'W': W, 'I': m_dict['I']}))

        # estimate interpolated onto the real (irregular) omega grid
        df_cmp = df_g[['P', 'W', 'Mean_M']].copy()
        df_cmp['Est_M'] = [np.interp(w, W, m_dict[f'M{int(p)}']) for p, w in zip(df_cmp.P, df_cmp.W)]
        df_cmp.sort_values(['P', 'W']).to_csv(os.path.join(DATA_DIR, f'M_real_vs_est_FM_{fm}.csv'), index=False)

        plot_real_vs_synthetic(df_g, est_full, fm)
        plot_real_vs_synthetic(df_g, est_full, fm, full_range=True)

        mean_abs = np.abs(df_cmp['Mean_M'] - df_cmp['Est_M']).mean()
        print(f'FM={fm}: {len(df_loss)} epochs, final loss_M={df_loss.loss_M.iloc[-1]:.3g}, '
              f'I_max={m_dict["I"].max():.4g}, mean|dM| vs real={mean_abs:.4g}')

    pd.concat(all_new_M, ignore_index=True)[['FM', 'P', 'W', 'Est_M']].to_csv(
        os.path.join(DATA_DIR, 'new_M.csv'), index=False)
    df_I = pd.concat(all_I, ignore_index=True)
    df_I.to_csv(os.path.join(DATA_DIR, 'I_omega_estimated.csv'), index=False)
    plot_I_omega(df_I)


def _fit_std_model(w_real, std_real):
    """Linear  std(omega) = slope*omega + intercept, as a callable of omega,
    floored at a tiny positive value so sigma stays valid."""
    slope, intercept = np.polyfit(w_real, std_real, 1)
    lo = 1e-6 * float(np.max(std_real))

    def model(w):
        return np.clip(slope * np.asarray(w, dtype=float) + intercept, lo, None)

    return model


def extrapolate_std_to_new_M(data=None, new_M=None):
    """Per (FM, P), fit the linear std(omega) model to the real data and evaluate
    it on new_M.csv's full omega grid (0..20).

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
    """N_fm = the largest real Exp count over P, for each FM."""
    if data is None:
        data = pd.read_csv(os.path.join(DATA_DIR, 'All_p_w_m.csv'))
    n_per_fm_p = data.groupby(['FM', 'P'])['Exp'].nunique()
    return n_per_fm_p.groupby(level='FM').max().to_dict()


def generate_synthetic_exp_data(new_M_S=None, N_fm=None, seed=12345):
    """For each (FM, P, W) draw N_fm[FM] replicas of M from Normal(Est_M, Est_S)
    truncated to M >= 0, then smooth each replica along omega.

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

        # truncnorm lower shape param in standardised units; sigma <= 0 rows get mu
        safe_sigma = np.where(sigma > 0, sigma, 1.0)
        a = (0.0 - mu) / safe_sigma
        samples = stats.truncnorm.rvs(a[:, None], np.inf, loc=mu[:, None],
                                       scale=safe_sigma[:, None],
                                       size=(len(g), n), random_state=rng)
        samples = np.where(sigma[:, None] > 0, samples, mu[:, None])

        # smooth each replica's own M(omega) curve within each P block (g is sorted P, W)
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
    out.loc[out.W == 0.0, 'M'] = 0.0
    out.to_csv(os.path.join(DATA_DIR, 'generated_Synthetic_exp_data.csv'), index=False)
    return out


def generate_m_omega_from_distribution(seed=12345):
    """new_M.csv's Est_M plus the extrapolated std(omega) as (mu, sigma) for N_fm
    synthetic Exp replicas per (FM, P, W)."""
    data = pd.read_csv(os.path.join(DATA_DIR, 'All_p_w_m.csv'))
    new_M_S = extrapolate_std_to_new_M(data)
    N_fm = max_n_exp_per_fm(data)
    print('N_fm (max real Exp count per FM):', N_fm)
    generate_synthetic_exp_data(new_M_S, N_fm, seed=seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()

    # build the real-data file from the raw rpitd files only if it is missing
    if not os.path.exists(os.path.join(DATA_DIR, 'All_p_w_m.csv')):
        create_combined_data()
    generate_m_omega()
    generate_m_omega_from_distribution(seed=args.seed)
