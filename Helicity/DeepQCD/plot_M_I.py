"""M(omega) for every P on one axis, and the Ioffe-time distribution I(omega) built from
them, as extrapolated by one DeepQCD model -- one column per FM.

For each FM every replica is seeded with its 4 synthetic points at omega = 0, 0.1s,
0.2s, 0.3s and rolled out to omega = 20 (common.rollout).  I(omega) follows the
Code/test_error.py rule: every momentum pair 2 <= p1 < p2,
    I = (p2^2 M(p2) - p1^2 M(p1)) / (p2^2 - p1^2),
averaged over the non-negative pair estimates at each omega (0 if all are negative).
The curve is I of the replica-mean M; its error is the jackknife error of the
per-replica I (jackknife.py).  The truth is the same rule applied to Est_M.
No Savitzky-Golay smoothing: the rolled-out curves are already smooth.

By default each FM uses the model trained without it (honest held-out curves);
--use_all uses the all-FM model.

Writes (../Data/DeepQCD/)
    M_all_P_{spec}_stride{s}.csv     FM, P, W, M_mean, M_err, Est_M
    I_omega_{spec}_stride{s}.csv     FM, W, I_mean, I_err, I_true
    Images/M_I_{spec}_stride{s}.png  row 1: M for every P, row 2: I, one column per FM

Run from Helicity/DeepQCD:  python plot_M_I.py [--model direct-noa-nolevel] [--stride 2]
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (FMS, FM_TO_A, IMG_DIR, N_SEED, OUT_DIR, P_SET, PLOT_DPI, get_device,
                    load_curves, load_model, model_path, rollout)
from jackknife import jk_error

MIN_P = 2                 # smallest P in an I(omega) pair, as in Code/test_error.py


I_MODES = ('clip', 'abs', 'signed')


def p_colors():
    """One colour per P (P_SET order): matplotlib's default cycle, as Code/test_error.py
    and Code/model_selection.py colour their per-P curves."""
    cyc = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return [cyc[k % len(cyc)] for k in range(len(P_SET))]


def i_omega(M, ps=P_SET, mode='clip'):
    """I(omega) from M of shape (nP, ..., n_w), row k being momentum ps[k], averaged over
    every momentum pair p >= MIN_P:
        clip    the test_error.py rule: mean of the non-negative pair estimates (0 if none)
        abs     mean of |pair estimate|
        signed  mean of the pair estimates as computed, negatives kept"""
    ps = np.asarray(ps)
    present = [k for k, p in enumerate(ps) if p >= MIN_P]
    E = np.stack([(ps[j] ** 2 * M[j] - ps[i] ** 2 * M[i]) / (ps[j] ** 2 - ps[i] ** 2)
                  for i in present for j in present if i < j])
    if mode == 'abs':
        return np.abs(E).mean(axis=0)
    if mode == 'signed':
        return E.mean(axis=0)
    with np.errstate(invalid='ignore'):
        return np.nan_to_num(np.nanmean(np.where(E >= 0, E, np.nan), axis=0), nan=0.0)


def ylim_from(ax, *curves):
    lo = min(np.nanmin(c) for c in curves)
    hi = max(np.nanmax(c) for c in curves)
    ax.set_ylim(lo - 0.15 * (hi - lo), hi + 0.25 * (hi - lo))


def main(args):
    device = get_device(args.device)
    curves = load_curves()
    drop = [d for d in args.model.split('-no')[1:]]
    head = args.model.split('-')[0]
    rows_M, rows_I = [], []
    fig, axes = plt.subplots(2, len(FMS), figsize=(6 * len(FMS), 9), squeeze=False)
    colors = p_colors()

    for c, fm in enumerate(FMS):
        path = model_path(None if args.use_all else fm, head, drop)
        model, _ = load_model(path, device)
        grid_i = np.arange(0, curves[fm]['M'].shape[2], args.stride)
        M = curves[fm]['M'][:, :, grid_i]                       # (nP, nExp, n_pts)
        nP, nE, n_pts = M.shape
        P = np.repeat(P_SET, nE)
        w, pred = rollout(model, np.full(len(P), FM_TO_A[fm]), P,
                          M[:, :, :N_SEED].reshape(nP * nE, N_SEED), args.stride, device)
        pred = pred.reshape(nP, nE, n_pts)
        est = curves[fm]['est'][:, grid_i]                      # (nP, n_pts)
        m_mean, m_err = pred.mean(axis=1), jk_error(pred, axis=1)

        I_mean = i_omega(m_mean)
        I_err = jk_error(i_omega(pred), axis=0)                 # per-replica I -> (nExp, n_pts)
        I_true = i_omega(est)

        for k, p in enumerate(P_SET):
            rows_M.append(pd.DataFrame({'FM': fm, 'P': p, 'W': np.round(w, 1), 'M_mean': m_mean[k],
                                        'M_err': m_err[k], 'Est_M': est[k]}))
        rows_I.append(pd.DataFrame({'FM': fm, 'W': np.round(w, 1), 'I_mean': I_mean,
                                    'I_err': I_err, 'I_true': I_true}))

        ax = axes[0, c]
        for k, p in enumerate(P_SET):
            ax.fill_between(w, m_mean[k] - m_err[k], m_mean[k] + m_err[k], color=colors[k], alpha=0.12, lw=0)
            ax.plot(w, est[k], '-', color=colors[k], lw=1.2, alpha=0.8)
            ax.plot(w, m_mean[k], '--', color=colors[k], lw=2, label=f'P={p}')
        ax.axvline(w[N_SEED - 1], color='0.6', lw=0.8, ls=':')
        ylim_from(ax, m_mean, est)
        ax.set_title(f'FM={fm} (a = {FM_TO_A[fm]} fm): M($\\omega$)')
        ax.set_xlabel(r'$\omega$')
        ax.legend(title='dashed: DNN, solid: truth', fontsize=9, title_fontsize=9)

        ax = axes[1, c]
        ax.fill_between(w, I_mean - I_err, I_mean + I_err, color='#2a78d6', alpha=0.2, lw=0)
        ax.plot(w, I_true, 'k-', lw=1.5, label='truth (from Est_M)')
        ax.plot(w, I_mean, '--', color='#2a78d6', lw=2, label='DNN')
        ax.axvline(w[N_SEED - 1], color='0.6', lw=0.8, ls=':')
        ylim_from(ax, I_mean, I_true)
        ax.set_title(f'FM={fm}: I($\\omega$), pairs $p_z \\geq {MIN_P}$')
        ax.set_xlabel(r'$\omega$')
        ax.legend(fontsize=9)
        rmse_m = np.sqrt(np.mean((m_mean[:, N_SEED:] - est[:, N_SEED:]) ** 2))
        rmse_i = np.sqrt(np.mean((I_mean[N_SEED:] - I_true[N_SEED:]) ** 2))
        print(f'FM={fm}: {os.path.basename(path)}  RMSE M {rmse_m:.5f}  RMSE I {rmse_i:.5f}')

    axes[0, 0].set_ylabel('M')
    axes[1, 0].set_ylabel('I')
    which = 'all-FM model' if args.use_all else 'held-out models'
    fig.suptitle(f'DeepQCD {args.model} ({which}), seeds at $\\omega \\leq$ '
                 f'{(N_SEED - 1) * args.stride * 0.1:g} (left of dotted line), bands = jackknife error')
    fig.tight_layout()
    tag = f'{args.model}{"_all" if args.use_all else ""}_stride{args.stride}'
    fig.savefig(os.path.join(IMG_DIR, f'M_I_{tag}.png'), dpi=PLOT_DPI)
    pd.concat(rows_M).to_csv(os.path.join(OUT_DIR, f'M_all_P_{tag}.csv'), index=False)
    pd.concat(rows_I).to_csv(os.path.join(OUT_DIR, f'I_omega_{tag}.csv'), index=False)
    print(f'wrote Images/M_I_{tag}.png, M_all_P_{tag}.csv, I_omega_{tag}.csv to {OUT_DIR}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='direct-noa-nolevel', help='model spec, e.g. direct-noa-nolevel, m3-noa')
    ap.add_argument('--stride', type=int, default=2, help='seed/grid spacing in 0.1 omega steps; must divide 200')
    ap.add_argument('--use_all', action='store_true', help='use the all-FM model instead of the held-out ones')
    ap.add_argument('--device', default='auto')
    main(ap.parse_args())
