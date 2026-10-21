"""Feed the real lattice data into the constrained joint-P network and plot M(omega) for
every P and I(omega), one column per FM.

Real data: ../Data/2026/All_p_w_m.csv, one row per (FM, P, Exp, Z) with omega = P z d1
(Z = 0 is the M = 0 anchor); each Exp row is a jackknife sample, the same gauge
configurations for every P, so replica Exp of every P is fed together.  The seeds are
Z = 0..3 of every P (common.rollout_lattice), and the network rolls out on the common
grid omega = k * dw up to 20.  Every output point obeys P^2 M_P rising in P and gives
one I(omega) for every pair, so I is the network's own I, with no clipping of negative
pair estimates needed.

The real points are compared with the DNN curve (interpolated to the real omega):
seeds (Z <= 3) and the check points never fed to the model (FM 9: Z = 4, 5; FM 12: Z = 4).
Because the constraint is imposed, the curve need not pass through seeds that break it;
the script counts how often the real data break it where two momenta share an omega
(P_i z_i = P_j z_j).

Writes (../Data/DeepQCDConstraint/)
    real_M_all_P_{tag}.csv       FM, P, W, M_mean, M_err, Est_M
    real_I_omega_{tag}.csv       FM, W, I_mean, I_err, I_Est_M
    real_vs_dnn_{tag}.csv        FM, P, Z, W, seed, real_mean, real_err, dnn_mean, dnn_err, pull
    images/real_M_I_{tag}_FM{fm}.png, images/real_I_all_FM_{tag}.png  (drawn by plot.py from the CSVs)

Run from Helicity/DeepQCDConstraint:  python plot_real_M_I.py [--dw 0.2] [--test_fm 9]
"""

import argparse
import os

import numpy as np
import pandas as pd

from common import (DATA_IN, FMS, N_SEED, OMEGA_MAX, OUT_DIR, P_SET, W_STEP,
                    get_device, i_from_pairs, interp_grid, load_curves, load_model, model_path,
                    rollout_lattice, run_tag, violation_rate)
from jackknife import jk_error
from plot import plot_real


def real_cube(d, fm):
    """Real M of one FM as (nExp, nP, nZ) and omega (nP, nZ)."""
    g = d[d.FM == fm]
    piv = g.pivot_table(index='Exp', columns=['P', 'Z'], values='M').sort_index(axis=1)
    if piv.isna().any().any():
        raise ValueError(f'FM={fm}: missing (Exp, P, Z) cells in the real data')
    nZ = g.Z.nunique()
    w = g.groupby(['P', 'Z']).W.mean().to_numpy().reshape(len(P_SET), nZ)
    return piv.to_numpy().reshape(len(piv), len(P_SET), nZ), w


def coincident_violations(M, w):
    """Real (P_i, P_j) pairs at the same omega: count and how many have
    i^2 M_i >= j^2 M_j on the jackknife mean."""
    m = M.mean(axis=0)
    n = bad = 0
    for i in range(len(P_SET)):
        for j in range(i + 1, len(P_SET)):
            for zi in range(1, w.shape[1]):
                zj = np.flatnonzero(np.isclose(w[j], w[i, zi]))
                if len(zj):
                    n += 1
                    bad += P_SET[i] ** 2 * m[i, zi] >= P_SET[j] ** 2 * m[j, zj[0]]
    return n, bad


def main(args):
    device = get_device(args.device)
    test_fm = int(args.test_fm) if args.test_fm else None
    path = model_path(test_fm, args.tag)
    model, cfg = load_model(path, device)
    dw = args.dw
    if not any(np.isclose(dw, s * W_STEP) for s in cfg['strides']):
        print(f'  dw = {dw} is not one of the trained strides {[s * W_STEP for s in cfg["strides"]]}')
    n_pts = int(round(OMEGA_MAX / dw)) + 1
    tag = run_tag(test_fm, args.tag) + f'_dw{dw:.1f}'
    real = pd.read_csv(os.path.join(DATA_IN, 'All_p_w_m.csv'))
    est_curves = load_curves()
    rows_M, rows_I, rows_c = [], [], []

    for fm in FMS:
        M, w_real = real_cube(real, fm)
        nE = len(M)
        w_seed = np.broadcast_to(w_real[:, :N_SEED], (nE, len(P_SET), N_SEED))
        w, pred = rollout_lattice(model, w_seed, M[:, :, :N_SEED], dw, n_pts, device)
        if not np.isfinite(pred).all():
            raise RuntimeError(f'FM={fm}: non-finite rollout')
        m_mean, m_err = pred.mean(axis=0), jk_error(pred, axis=0)
        I_rep = i_from_pairs(np.moveaxis(pred, 1, 0))
        I_mean, I_err = I_rep.mean(axis=0), jk_error(I_rep, axis=0)
        est = interp_grid(est_curves[fm]['est'], np.broadcast_to(w, (len(P_SET), n_pts)))
        I_est = i_from_pairs(est)
        n_co, bad_co = coincident_violations(M, w_real)
        print(f'FM={fm}: output violation rate {violation_rate(pred[:, :, 1:]):.3g}; real data break the '
              f'constraint at {bad_co} of {n_co} coincident-omega momentum pairs')

        for k, p in enumerate(P_SET):
            rows_M.append(pd.DataFrame({'FM': fm, 'P': p, 'W': np.round(w, 1), 'M_mean': m_mean[k], 'M_err': m_err[k],
                                        'Est_M': est[k]}))
            for z in range(1, w_real.shape[1]):
                dnn = np.array([np.interp(w_real[k, z], w, row) for row in pred[:, k]])
                r_mean, r_err, d_mean, d_err = M[:, k, z].mean(), jk_error(M[:, k, z]), dnn.mean(), jk_error(dnn)
                rows_c.append({'FM': fm, 'P': p, 'Z': z, 'W': w_real[k, z], 'seed': z < N_SEED,
                               'real_mean': r_mean, 'real_err': r_err, 'dnn_mean': d_mean, 'dnn_err': d_err,
                               'pull': (d_mean - r_mean) / np.hypot(r_err, d_err)})
        rows_I.append(pd.DataFrame({'FM': fm, 'W': np.round(w, 1), 'I_mean': I_mean, 'I_err': I_err, 'I_Est_M': I_est}))

    pd.concat(rows_M).to_csv(os.path.join(OUT_DIR, f'real_M_all_P_{tag}.csv'), index=False)
    pd.concat(rows_I).to_csv(os.path.join(OUT_DIR, f'real_I_omega_{tag}.csv'), index=False)
    dfc = pd.DataFrame(rows_c)
    dfc.to_csv(os.path.join(OUT_DIR, f'real_vs_dnn_{tag}.csv'), index=False)
    plot_real(tag, args.xmax)
    pd.set_option('display.width', 200)
    print('\nreal points vs constrained DNN (pull = (dnn - real) / combined jackknife error)')
    print(dfc.round(4).to_string(index=False))
    print(f'wrote images/real_M_I_{tag}_FM*.png, real_M_all_P_{tag}.csv, real_I_omega_{tag}.csv, '
          f'real_vs_dnn_{tag}.csv to {OUT_DIR}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_fm', default='', help='use that leave-out checkpoint instead of the all-FM one')
    ap.add_argument('--tag', default='')
    ap.add_argument('--dw', type=float, default=0.2, help='rollout grid spacing (a trained stride x 0.1)')
    ap.add_argument('--xmax', type=float, default=20.0, help='omega range shown in the plots')
    ap.add_argument('--device', default='auto')
    main(ap.parse_args())
