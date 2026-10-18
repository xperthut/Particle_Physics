"""Score each leave-one-FM-out model on its held-out FM, window by window.

For every window (P, Exp, stride, start) of the held-out FM the 4 seeds are given
and the 6 next M values are predicted by
    DeepQCD_{spec} every model trained without this FM (train.py --test_fm FM ...)
    M3_seed        M = A w exp(-s w) fit to the seeds (common.m3_seed_fit)
    Lag_XGB_FM{y}  the existing 4-lag XGB models of training FM y
                   (../model/2026/{y}/XGB_exp_No.sav), rolled 6 one-step predictions;
                   stride 1 only, since they were trained on the 0.1 grid
and compared with the replica's own M ('rep', noisy) and the noise-free Est_M
('clean').  RMSE is over windows; rel_err = mean |pred - Est_M| / Est_M.

Writes (../Data/DeepQCD/)
    eval_by_step.csv     per (test_fm, method, stride, step 1..6)
    eval_by_P.csv        per (test_fm, method, stride, P)
    eval_summary.csv     per (test_fm, method, stride)
    Images/eval_step_FM{fm}.png   clean RMSE vs steps ahead, one panel per stride

Run from Helicity/DeepQCD:  python evaluate.py [--test_fm 15] [--max_exp 200]
"""

import argparse
import os
import warnings

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (FMS, FM_TO_A, IMG_DIR, LAG_MODEL_DIR, N_PRED, N_SEED, N_W, OUT_DIR,
                    P_SET, PLOT_DPI, W_STEP, WIN, find_models, get_device, load_curves,
                    load_model, m3_seed_fit, predict_m)

warnings.filterwarnings('ignore')

LAG_FEATURES = ['P', 'w1', 'm1', 'w2', 'm2', 'w3', 'm3', 'w4', 'm4', 'w5']   # Code/reconstruct_real_data.py


def all_windows(curves_fm, stride, exps):
    """Every window of one FM at one stride: P, w (n, WIN), m (n, WIN), Est_M (n, WIN)."""
    starts = np.arange(N_W - (WIN - 1) * stride)
    pi, ei, si = (g.ravel() for g in np.meshgrid(np.arange(len(P_SET)), exps, starts, indexing='ij'))
    idx = si[:, None] + stride * np.arange(WIN)
    return (P_SET[pi], idx * W_STEP, curves_fm['M'][pi[:, None], ei[:, None], idx],
            curves_fm['est'][pi[:, None], idx])


def load_lag_models(fm, name):
    path = os.path.join(LAG_MODEL_DIR, str(fm), f'{name}_exp_No.sav')
    try:
        return dict(zip(P_SET, joblib.load(path)))
    except Exception as e:                 # e.g. pickled with an incompatible sklearn
        print(f'  {name} FM={fm}: cannot load {path} ({type(e).__name__}: {e}), skipped')
        return None


def lag_roll(models, P, w, m_seed):
    m = np.concatenate([m_seed, np.empty((len(P), N_PRED))], axis=1)
    for k in range(N_SEED, WIN):
        lags = np.empty((len(P), 2 * N_SEED))
        lags[:, 0::2], lags[:, 1::2] = w[:, k - N_SEED:k], m[:, k - N_SEED:k]
        X = pd.DataFrame(np.column_stack([P, lags, w[:, k]]), columns=LAG_FEATURES)
        for p in P_SET:
            s = P == p
            m[s, k] = models[p].predict(X[s])
    return m[:, N_SEED:]


def scores(test_fm, method, stride, P, pred, rep, clean):
    err_r, err_c = pred - rep, pred - clean
    rel = np.abs(err_c) / clean
    base = {'test_fm': test_fm, 'method': method, 'stride': stride}
    by_step = [{**base, 'step': k + 1,
                'rmse_rep': np.sqrt(np.mean(err_r[:, k] ** 2)),
                'rmse_clean': np.sqrt(np.mean(err_c[:, k] ** 2)),
                'rel_err': rel[:, k].mean()} for k in range(N_PRED)]
    by_p = [{**base, 'P': p,
             'rmse_rep': np.sqrt(np.mean(err_r[P == p] ** 2)),
             'rmse_clean': np.sqrt(np.mean(err_c[P == p] ** 2)),
             'rel_err': rel[P == p].mean()} for p in P_SET]
    summ = {**base, 'n_windows': len(P), 'rmse_rep': np.sqrt(np.mean(err_r ** 2)),
            'rmse_clean': np.sqrt(np.mean(err_c ** 2)), 'rel_err': rel.mean()}
    return by_step, by_p, summ


def plot_steps(df, test_fm):
    strides = sorted(df.stride.unique())
    fig, axes = plt.subplots(1, len(strides), figsize=(4.5 * len(strides), 4), squeeze=False, sharey=True)
    for ax, s in zip(axes.flat, strides):
        for method, g in df[df.stride == s].groupby('method'):
            ax.semilogy(g.step, g.rmse_clean, 'o-', label=method)
        ax.set_title(f'stride {s} ($\\Delta\\omega$ = {s * W_STEP:g})')
        ax.set_xlabel('steps ahead')
    axes.flat[0].set_ylabel('RMSE vs Est_M')
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f'held-out FM={test_fm}')
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, f'eval_step_FM{test_fm}.png'), dpi=PLOT_DPI)
    plt.close(fig)


def main(args):
    device = get_device(args.device)
    curves = load_curves()
    specs = args.models.split(',') if args.models else None
    test_fms = [int(f) for f in args.test_fm.split(',')] if args.test_fm else \
        [fm for fm in FMS if find_models(fm, specs)]
    rng = np.random.default_rng(args.seed)
    step_rows, p_rows, summ_rows = [], [], []

    for test_fm in test_fms:
        models = {k: load_model(p, device) for k, p in find_models(test_fm, specs).items()}
        if not models:
            print(f'held-out FM={test_fm}: no checkpoints, skipped')
            continue
        cfg = next(iter(models.values()))[1]
        strides = [int(s) for s in args.strides.split(',')] if args.strides else cfg['strides']
        n_exp = curves[test_fm]['M'].shape[1]
        exps = np.sort(rng.choice(n_exp, args.max_exp, replace=False)) \
            if 0 < args.max_exp < n_exp else np.arange(n_exp)
        lag = {} if args.lag_models == 'none' else {
            f'Lag_{name}_FM{fm}': lag_m for name in args.lag_models.split(',')
            for fm in cfg['train_fms'] if (lag_m := load_lag_models(fm, name)) is not None}
        print(f'held-out FM={test_fm}: {list(models)} trained on {cfg["train_fms"]}, {len(exps)} replicas')

        for stride in strides:
            P, w, m, est = all_windows(curves[test_fm], stride, exps)
            seed, rep, clean = m[:, :N_SEED], m[:, N_SEED:], est[:, N_SEED:]
            a = np.full(len(P), FM_TO_A[test_fm])
            preds = {name: predict_m(mdl, a, P, w, seed, device) for name, (mdl, _) in models.items()}
            preds['M3_seed'] = m3_seed_fit(w[:, :N_SEED], seed, w[:, N_SEED:])
            if stride == 1:
                for name, lag_m in lag.items():
                    preds[name] = lag_roll(lag_m, P, w, seed)
            for method, pred in preds.items():
                a, b, c = scores(test_fm, method, stride, P, pred, rep, clean)
                step_rows += a
                p_rows += b
                summ_rows.append(c)
                print(f'  stride {stride:2d}  {method:16s}  RMSE clean {c["rmse_clean"]:.5f}  '
                      f'rep {c["rmse_rep"]:.5f}  rel {c["rel_err"]:.4f}')
        plot_steps(pd.DataFrame([r for r in step_rows if r['test_fm'] == test_fm]), test_fm)

    pd.DataFrame(step_rows).to_csv(os.path.join(OUT_DIR, 'eval_by_step.csv'), index=False)
    pd.DataFrame(p_rows).to_csv(os.path.join(OUT_DIR, 'eval_by_P.csv'), index=False)
    pd.DataFrame(summ_rows).to_csv(os.path.join(OUT_DIR, 'eval_summary.csv'), index=False)
    print(f'wrote eval_by_step.csv, eval_by_P.csv, eval_summary.csv to {OUT_DIR}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_fm', default='', help='comma list; default: every FM with a saved model')
    ap.add_argument('--models', default='', help='comma list of model specs (e.g. m3,m3-noa); default: every checkpoint')
    ap.add_argument('--strides', default='', help='comma list; default: the strides the model trained on')
    ap.add_argument('--max_exp', type=int, default=0, help='random subset of replicas (0 = all)')
    ap.add_argument('--lag_models', default='XGB', help="comma list of Code/ lag models, or 'none'")
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--device', default='auto')
    main(ap.parse_args())
