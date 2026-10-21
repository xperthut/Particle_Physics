"""Score the constrained joint-P network on the FM it was not trained on, against the
noise-free Est_M, next to the unconstrained per-P DeepQCD model (direct-noa-nolevel).

1. Windows: 4 seeds -> 6 targets, both seed layouts, every stride.  RMSE per P and the
   constraint-violation rate (cells where P^2 M_P does not rise strictly with P).
2. Full rollouts to omega = 20, from every replica of the held-out FM:
     grid     seeds at omega = 0, dw, 2dw, 3dw for every P
     lattice  seeds at that FM's real lattice omegas P z d1, z = 0..3 (what the real data
              give); DeepQCD rolls each P out on its own grid P d1 and is interpolated
              onto the common grid, as DeepQCD/plot_real_M_I.py does
   RMSE of the replica-mean M (extrapolated points, averaged over P) and of I(omega),
   I being the signed mean over every pair 2 <= p_i < p_j (Code/test_error.py pairs),
   and the violation rate over every replica and omega > 0.

Writes (../Data/DeepQCDConstraint/)
    eval_windows.csv    test_fm, model, layout, stride, P, rmse, violation_rate
    eval_rollout.csv    test_fm, model, layout, dw, rmse_M, rmse_I, violation_rate
    images/rollout_{layout}_dw{dw}_FM{fm}.png   M per P | I, one image per held-out FM,
                                                only for the --plots settings (default lattice:0.5)
    images/rollout_{layout}_dw{dw}_I_all_FM.png, and for the same settings
        rollout_M_all_P_{layout}_dw{dw}.csv  FM, P, W, M_mean, M_err, Est_M
        rollout_seeds_{layout}_dw{dw}.csv    FM, P, W, M_mean (seed points)
        rollout_I_omega_{layout}_dw{dw}.csv  FM, W, I_mean, I_err, I_true, I_DeepQCD (-> continuum_I.py)
    images are drawn by plot.py from these CSVs

Run from Helicity/DeepQCDConstraint:  python evaluate.py
"""

import argparse
import importlib.util
import os

import numpy as np
import pandas as pd

from common import (DATA_IN, FMS, FM_TO_A, N_SEED, N_W, OMEGA_MAX, OUT_DIR, P_SET,
                    W_STEP, get_device, i_from_pairs, interp_grid, load_curves, load_model, model_path,
                    predict, rollout_grid, rollout_lattice, sample_windows, violation_rate)
from jackknife import jk_error
from plot import plot_rollout

OLD_SPEC = 'direct-noa-nolevel'


def load_deepqcd():
    """DeepQCD's common.py under another module name (both folders call it common)."""
    spec = importlib.util.spec_from_file_location('deepqcd_common', '../DeepQCD/common.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def real_d1():
    """{fm: omega of P = 1, z = 1} of the real lattice data."""
    d = pd.read_csv(os.path.join(DATA_IN, 'All_p_w_m.csv'), usecols=['FM', 'P', 'Z', 'W'])
    return d[(d.P == 1) & (d.Z == 1)].groupby('FM').W.mean().to_dict()


def old_windows(old, model, fm, w_seed, m_seed, w_tgt, device):
    """DeepQCD per-P prediction of grid windows, (n, nP, 6)."""
    n = len(m_seed)
    out = np.empty((n, len(P_SET), w_tgt.shape[1]))
    w_win = np.concatenate([w_seed[:, 0], w_tgt], axis=1)
    for k, p in enumerate(P_SET):
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            out[:, k] = old.predict_m(model, np.full(n, FM_TO_A[fm]), np.full(n, p), w_win, m_seed[:, k], device)
    return out


def old_rollout(old, model, fm, w_seed, m_seed, w_common, device):
    """DeepQCD per-P rollouts on each P's own seed spacing, interpolated onto w_common."""
    n = len(m_seed)
    out = np.empty((n, len(P_SET), len(w_common)))
    for k, p in enumerate(P_SET):
        dw = w_seed[0, k, 1]
        n_pts = int(np.floor(OMEGA_MAX / dw + 1e-9)) + 1
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            w, pred = old.rollout_dw(model, np.full(n, FM_TO_A[fm]), np.full(n, p), m_seed[:, k], dw, n_pts, device)
        out[:, k] = np.array([np.interp(w_common, w, row, right=np.nan) for row in pred])
    return out


def rmse(a, b):
    return float(np.sqrt(np.nanmean((a - b) ** 2)))


def main(args):
    device = get_device(args.device)
    curves = load_curves()
    old = load_deepqcd()
    d1 = real_d1()
    strides = [int(s) for s in args.strides.split(',')]
    rows_w, rows_r = [], []
    plots = {}

    for fm in FMS:
        path = model_path(fm)
        if not os.path.exists(path):
            print(f'FM={fm}: no {path}, skipped')
            continue
        model, _ = load_model(path, device)
        old_path = os.path.join(old.MODEL_DIR, f'deepqcd_{OLD_SPEC}_test_FM{fm}.pt')
        old_model = old.load_model(old_path, device)[0] if os.path.exists(old_path) else None
        rng = np.random.default_rng(args.seed)
        est = curves[fm]['est']
        reps = curves[fm]['M']                                       # (nP, nExp, N_W)
        nE = reps.shape[1]

        # 1. windows
        for layout in ('grid', 'lattice'):
            for st in strides:
                w_seed, m_seed, w_tgt, _, est_tgt = sample_windows(
                    curves, [fm], args.n_windows, [st], rng, lattice_frac=float(layout == 'lattice'))
                preds = {'DeepQCDConstraint': predict(model, w_seed, m_seed, w_tgt, device)[0]}
                if layout == 'grid' and old_model is not None:
                    preds[f'DeepQCD_{OLD_SPEC}'] = old_windows(old, old_model, fm, w_seed, m_seed, w_tgt, device)
                for name, pred in preds.items():
                    vr = violation_rate(pred)
                    for k, p in enumerate(P_SET):
                        rows_w.append({'test_fm': fm, 'model': name, 'layout': layout, 'stride': st, 'P': p,
                                       'rmse': rmse(pred[:, k], est_tgt[:, k]), 'violation_rate': vr})

        # 2. full rollouts from every replica
        for layout in ('grid', 'lattice'):
            for st in strides:
                dw = st * W_STEP
                n_pts = int(round(OMEGA_MAX / dw)) + 1
                w = np.arange(n_pts) * dw
                if layout == 'grid':
                    w_seed = np.broadcast_to(w[:N_SEED], (nE, len(P_SET), N_SEED))
                    m_seed = np.transpose(reps[:, :, (np.arange(N_SEED) * st)], (1, 0, 2))
                    _, M = rollout_grid(model, m_seed, dw, n_pts, device)
                    first = N_SEED
                else:
                    w_seed = np.broadcast_to((P_SET[:, None] * np.arange(N_SEED)[None]) * d1[fm],
                                             (nE, len(P_SET), N_SEED))
                    m_seed = interp_grid(np.transpose(reps, (1, 0, 2)), w_seed)
                    _, M = rollout_lattice(model, w_seed, m_seed, dw, n_pts, device)
                    first = 1
                est_w = interp_grid(est, np.broadcast_to(w, (len(P_SET), n_pts)))
                I_true = i_from_pairs(est_w)
                res = {'DeepQCDConstraint': M}
                if old_model is not None:
                    res[f'DeepQCD_{OLD_SPEC}'] = old_rollout(old, old_model, fm, w_seed, m_seed, w, device)
                for name, R in res.items():
                    R = R.copy()
                    R[~np.isfinite(R)] = np.nan
                    m_mean = np.nanmean(R, axis=0)                     # (nP, n_pts)
                    I_rep = i_from_pairs(np.moveaxis(R, 1, 0))         # (nExp, n_pts)
                    rows_r.append({'test_fm': fm, 'model': name, 'layout': layout, 'dw': round(dw, 1),
                                   'rmse_M': rmse(m_mean[:, first:], est_w[:, first:]),
                                   'rmse_I': rmse(i_from_pairs(m_mean)[first:], I_true[first:]),
                                   'violation_rate': violation_rate(np.nan_to_num(R[:, :, 1:], nan=0.0))})
                    plots.setdefault((layout, st), {}).setdefault(fm, {})[name] = (
                        w, m_mean, jk_error(R, axis=0), np.nanmean(I_rep, axis=0), jk_error(I_rep, axis=0),
                        est_w, I_true, m_seed, w_seed)
        print(f'FM={fm} done')

    dfw, dfr = pd.DataFrame(rows_w), pd.DataFrame(rows_r)
    dfw.to_csv(os.path.join(OUT_DIR, 'eval_windows.csv'), index=False)
    dfr.to_csv(os.path.join(OUT_DIR, 'eval_rollout.csv'), index=False)
    pd.set_option('display.width', 200)
    print('\nwindow RMSE vs Est_M (mean over P) and violation rate')
    print(dfw.groupby(['test_fm', 'layout', 'stride', 'model'])[['rmse', 'violation_rate']].mean()
          .unstack('model').round(4).to_string())
    print('\nfull rollout to omega = 20')
    print(dfr.set_index(['test_fm', 'layout', 'dw', 'model'])[['rmse_M', 'rmse_I', 'violation_rate']]
          .unstack('model').round(4).to_string())

    wanted = {(lay, int(round(float(dw) / W_STEP))) for lay, dw in (p.split(':') for p in args.plots.split(','))}
    for (layout, st), per_fm in plots.items():
        if (layout, st) not in wanted:
            continue
        rows_M, rows_S, rows_I = [], [], []
        for fm, res in per_fm.items():
            w, m_mean, m_err, I_mean, I_err, est_w, I_true, m_seed, w_seed = res['DeepQCDConstraint']
            for k, p in enumerate(P_SET):
                rows_M.append(pd.DataFrame({'FM': fm, 'P': p, 'W': np.round(w, 1), 'M_mean': m_mean[k],
                                            'M_err': m_err[k], 'Est_M': est_w[k]}))
                rows_S.append(pd.DataFrame({'FM': fm, 'P': p, 'W': w_seed[0, k], 'M_mean': m_seed[:, k].mean(axis=0)}))
            r = pd.DataFrame({'FM': fm, 'W': np.round(w, 1), 'I_mean': I_mean, 'I_err': I_err, 'I_true': I_true})
            if f'DeepQCD_{OLD_SPEC}' in res:
                r['I_DeepQCD'] = res[f'DeepQCD_{OLD_SPEC}'][3]
            rows_I.append(r)
        key = f'{layout}_dw{st * W_STEP:.1f}'
        pd.concat(rows_M).to_csv(os.path.join(OUT_DIR, f'rollout_M_all_P_{key}.csv'), index=False)
        pd.concat(rows_S).to_csv(os.path.join(OUT_DIR, f'rollout_seeds_{key}.csv'), index=False)
        pd.concat(rows_I).to_csv(os.path.join(OUT_DIR, f'rollout_I_omega_{key}.csv'), index=False)
        plot_rollout(layout, f'{st * W_STEP:.1f}')
    print(f'wrote eval_windows.csv, eval_rollout.csv, images/rollout_*_FM*.png to {OUT_DIR}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--strides', default='1,2,5,10')
    ap.add_argument('--plots', default='lattice:0.5',
                    help="comma list of layout:dw to plot, e.g. 'lattice:0.5,grid:1.0' (all are scored)")
    ap.add_argument('--n_windows', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='auto')
    main(ap.parse_args())
