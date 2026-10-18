"""Extrapolate whole M(omega) curves to omega = 20 from 4 small-omega seeds -- the way
the model would be used on a new ensemble.

For each FM, each replica's seeds are its synthetic M at omega = 0, 0.1s, 0.2s, 0.3s
(stride s); common.rollout predicts 6 points per block and re-seeds from its own
last 4 predictions until omega = 20.  The M3_seed baseline fits A w exp(-s w) to the
same seeds and evaluates it on the whole grid.  Means and jackknife errors
(sqrt(N-1) std over replicas, jackknife.py) are compared with the noise-free Est_M.

By default FM uses the models trained without it (deepqcd_{head}_test_FM{fm}.pt), so
every curve is an honest held-out extrapolation; --use_all uses the all-FM models
(deepqcd_{head}_all.pt), the ones meant for new ensembles.

Writes (../Data/DeepQCD/)
    extrapolated_M_FM{fm}_stride{s}.csv   FM, P, W, Est_M, Rep_mean/err, DeepQCD_{head}_mean/err, M3_seed_mean/err
    extrapolation_summary.csv             RMSE / max |error| of the mean curve vs Est_M per (FM, stride, method, P)
    Images/extrapolated_M_FM{fm}_stride{s}.png

Run from Helicity/DeepQCD:  python extrapolate.py [--fm 15] [--strides 5,10]
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (FMS, FM_TO_A, IMG_DIR, N_SEED, OUT_DIR, P_SET, PLOT_DPI, find_models,
                    get_device, load_curves, load_model, m3_seed_fit, rollout)
from jackknife import jk_error


def plot_fm(df, fm, stride, methods, model_name):
    fig, axes = plt.subplots(1, len(P_SET), figsize=(4.2 * len(P_SET), 4))
    for ax, p in zip(axes, P_SET):
        g = df[df.P == p]
        # y-range from the curves, not the error bands (an unstable fit's band can be huge)
        lines = g[['Est_M'] + [f'{m}_mean' for m in methods]].to_numpy()
        lo, hi = np.nanmin(lines), np.nanmax(lines)
        ax.set_ylim(lo - 0.25 * (hi - lo), hi + 0.25 * (hi - lo))
        ax.plot(g.W, g.Est_M, 'k-', lw=1.5, label='Est_M (truth)')
        for method, col in zip(methods, plt.cm.tab10.colors):
            ax.plot(g.W, g[f'{method}_mean'], '--', color=col, label=method)
            ax.fill_between(g.W, g[f'{method}_mean'] - g[f'{method}_err'],
                            g[f'{method}_mean'] + g[f'{method}_err'], color=col, alpha=0.2, lw=0)
        ax.axvline(g.W.iloc[N_SEED - 1], color='0.6', lw=0.8, ls=':')
        ax.set_title(f'P={p}')
        ax.set_xlabel(r'$\omega$')
    axes[0].set_ylabel('M')
    axes[0].legend(fontsize=8)
    fig.suptitle(f'FM={fm}, stride {stride}, {model_name} models (seeds left of dotted line)')
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, f'extrapolated_M_FM{fm}_stride{stride}.png'), dpi=PLOT_DPI)
    plt.close(fig)


def main(args):
    device = get_device(args.device)
    curves = load_curves()
    fms = [int(f) for f in args.fm.split(',')] if args.fm else FMS
    rng = np.random.default_rng(args.seed)
    summary = []

    model_name = 'all-FM' if args.use_all else 'leave-out'
    for fm in fms:
        paths = find_models(None if args.use_all else fm, args.models.split(',') if args.models else None)
        models = {k: load_model(p, device) for k, p in paths.items()}
        if not models:
            print(f'FM={fm}: no {model_name} checkpoints found -- run train.py first')
            continue
        cfg = next(iter(models.values()))[1]
        if fm in cfg['train_fms']:
            print(f'FM={fm}: note, the {model_name} models were trained on this FM (not a held-out test)')
        methods = list(models) + ['M3_seed']
        strides = [int(s) for s in args.strides.split(',')] if args.strides else cfg['strides']
        M_all, est = curves[fm]['M'], curves[fm]['est']
        n_exp = M_all.shape[1]
        exps = np.sort(rng.choice(n_exp, args.max_exp, replace=False)) \
            if 0 < args.max_exp < n_exp else np.arange(n_exp)

        for stride in strides:
            grid_i = np.arange(0, M_all.shape[2], stride)
            M = M_all[:, exps][:, :, grid_i]                      # (nP, nExp, n_pts)
            nP, nE, n_pts = M.shape
            P = np.repeat(P_SET, nE)
            seed = M[:, :, :N_SEED].reshape(nP * nE, N_SEED)
            per = {'Rep': M}
            for name, (mdl, _) in models.items():
                w, deep = rollout(mdl, np.full(len(P), FM_TO_A[fm]), P, seed, stride, device)
                per[name] = deep.reshape(nP, nE, n_pts)
            m3 = m3_seed_fit(np.broadcast_to(w[:N_SEED], seed.shape), seed,
                             np.broadcast_to(w, (len(P), n_pts)))
            per['M3_seed'] = m3.reshape(nP, nE, n_pts)

            df = pd.DataFrame({'FM': fm, 'P': np.repeat(P_SET, n_pts), 'W': np.tile(np.round(w, 1), nP),
                               'Est_M': est[:, grid_i].ravel()})
            for name, arr in per.items():
                df[f'{name}_mean'] = arr.mean(axis=1).ravel()
                df[f'{name}_err'] = jk_error(arr, axis=1).ravel()
            df.to_csv(os.path.join(OUT_DIR, f'extrapolated_M_FM{fm}_stride{stride}.csv'), index=False)
            plot_fm(df, fm, stride, methods, model_name)

            beyond = df.W > w[N_SEED - 1]                         # score only extrapolated points
            for method in methods:
                for p in P_SET:
                    g = df[beyond & (df.P == p)]
                    e = g[f'{method}_mean'] - g.Est_M
                    summary.append({'FM': fm, 'model': model_name, 'stride': stride, 'method': method,
                                    'P': p, 'rmse_mean_vs_Est_M': np.sqrt(np.mean(e ** 2)),
                                    'max_abs_err': np.abs(e).max(),
                                    'mean_jk_err': g[f'{method}_err'].mean()})
            s = pd.DataFrame(summary)
            s = s[(s.FM == fm) & (s.stride == stride)].groupby('method').rmse_mean_vs_Est_M.mean()
            print(f'FM={fm} stride {stride:2d}: ' +
                  '  '.join(f'{k} RMSE {v:.5f}' for k, v in s.items()))

    pd.DataFrame(summary).to_csv(os.path.join(OUT_DIR, 'extrapolation_summary.csv'), index=False)
    print(f'wrote extrapolation_summary.csv to {OUT_DIR}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fm', default='', help='comma list; default: all FMs')
    ap.add_argument('--models', default='', help='comma list of model specs (e.g. m3,m3-noa); default: every checkpoint')
    ap.add_argument('--use_all', action='store_true', help='use the all-FM models instead of the leave-out ones')
    ap.add_argument('--strides', default='', help='comma list; each must divide 200 (default: training strides)')
    ap.add_argument('--max_exp', type=int, default=0, help='random subset of replicas (0 = all)')
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--device', default='auto')
    main(ap.parse_args())
