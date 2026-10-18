"""Feed the real lattice data straight into a DeepQCD model and plot M(omega) for every P
and the I(omega) built from them, one column per FM (layout as plot_M_I.py).

Real data: ../Data/2026/All_p_w_m.csv, one row per (FM, P, Exp, Z) with omega = dw_P * Z
(evenly spaced, dw_P depends on FM and P; Z = 0 is the M = 0 anchor).  Each Exp row is a
jackknife sample.  For every (FM, P, Exp) the first 4 real points (Z = 0..3) are the
seeds and common.rollout_dw extrapolates on that P's own grid omega = dw_P * k up to
omega = 20.  Real points past the seeds (FM 9: Z = 4, 5; FM 12: Z = 4; FM 15: none) are
never fed to the model and are plotted as a check.

The per-replica curves are interpolated onto the common grid omega = 0, 0.1, ..
(up to the smallest last grid point over P) so the momentum pairs line up; I(omega)
uses plot_M_I.i_omega (Code/test_error.py pair rule, p >= 2).  Errors are jackknife
errors over the Exp rows.

A (FM, P) whose rollout is unstable -- any replica non-finite or rising above
BLOWUP x its largest seed (happens when the last seed is ~0, since the network works
relative to it) -- is flagged: its real points are still drawn, but it gets no DNN
curve and is left out of that FM's I(omega) pairs.  Dropping only the bad replicas
would bias the jackknife.  The generator's noise-free curve Est_M and its I(omega) are
drawn faintly for reference -- they are a model fit, not lattice data.

Writes (../Data/DeepQCD/)
    real_M_all_P_{spec}.csv     FM, P, W, M_mean, M_err
    real_I_omega_{spec}.csv     FM, W, I_mean, I_err, I_Est_M
    Images/real_M_I_{spec}.png

Run from Helicity/DeepQCD:  python plot_real_M_I.py [--model direct-noa-nolevel-wide]
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (DATA_IN, FMS, FM_TO_A, IMG_DIR, N_SEED, OMEGA_MAX, OUT_DIR, P_SET,
                    PLOT_DPI, W_STEP, find_models, get_device, load_curves, load_model,
                    rollout_dw)
from jackknife import jk_error
from plot_M_I import I_MODES, MIN_P, i_omega, p_colors, ylim_from

BLOWUP = 5.0              # a replica rising above BLOWUP x its largest seed marks the rollout unstable
MODE_LABEL = {'clip': '', 'abs': ', mean |pair I|', 'signed': ', mean pair I (signed)'}
MODE_TITLE = {'clip': '',
              'abs': '\nI = mean over pairs of |pair estimate| (the real pairs are negative)',
              'signed': '\nI = mean over pairs of the pair estimate as computed (negative values kept)'}


def real_cube(d, fm, p):
    """Real M of one (FM, P) as (nExp, nZ), and the omegas of the Z columns."""
    g = d[(d.FM == fm) & (d.P == p)]
    piv = g.pivot_table(index='Exp', columns='Z', values='M').sort_index()
    w = g.groupby('Z').W.mean().reindex(piv.columns).to_numpy()
    if piv.isna().any().any():
        raise ValueError(f'FM={fm} P={p}: missing (Exp, Z) cells in the real data')
    return piv.to_numpy(), w


def main(args):
    device = get_device(args.device)
    paths = find_models(None if not args.test_fm else int(args.test_fm), [args.model])
    if not paths:
        raise FileNotFoundError(f'no checkpoint for spec {args.model} -- run train.py first')
    path = next(iter(paths.values()))
    model, cfg = load_model(path, device)
    max_dw = max(cfg['strides']) * W_STEP
    real = pd.read_csv(os.path.join(DATA_IN, 'All_p_w_m.csv'))
    est_curves = load_curves()
    colors = p_colors()
    rows_M, rows_I = [], []
    fig, axes = plt.subplots(2, len(FMS), figsize=(6 * len(FMS), 9), squeeze=False)

    for c, fm in enumerate(FMS):
        per_p = {}
        for p in P_SET:
            M, w_real = real_cube(real, fm, p)
            if (M[:, N_SEED - 1] <= 0).any():          # the network works relative to the last seed
                raise ValueError(f'FM={fm} P={p}: non-positive M at the last seed (Z={N_SEED - 1})')
            dw = w_real[1]
            if dw > max_dw + 1e-9:
                print(f'  FM={fm} P={p}: spacing {dw:.3f} beyond the trained {max_dw:g}')
            n_pts = int(np.floor(OMEGA_MAX / dw + 1e-9)) + 1
            with np.errstate(over='ignore', invalid='ignore'):
                w, pred = rollout_dw(model, np.full(len(M), FM_TO_A[fm]), np.full(len(M), p),
                                     M[:, :N_SEED], dw, n_pts, device)
            bad = ~np.isfinite(pred).all(axis=1) | (np.nanmax(np.where(np.isfinite(pred), pred, np.nan), axis=1)
                                                    > BLOWUP * M[:, :N_SEED].max(axis=1))
            if bad.any():
                print(f'  FM={fm} P={p}: rollout unstable for {bad.sum()} of {len(M)} replicas '
                      f'-> no DNN curve, left out of I(omega)')
            per_p[p] = (w, pred, M, w_real, bad.any())

        good = [p for p in P_SET if not per_p[p][4]]
        w_common = np.round(np.arange(0, min(per_p[p][0][-1] for p in good) + 1e-9, W_STEP), 1)
        cube = np.stack([np.array([np.interp(w_common, per_p[p][0], row) for row in per_p[p][1]])
                         for p in good])                          # (n_good, nExp, n_common)
        m_mean, m_err = cube.mean(axis=1), jk_error(cube, axis=1)
        I_mean = i_omega(m_mean, good, args.I_mode)
        I_err = jk_error(i_omega(cube, good, args.I_mode), axis=0)
        est_all = est_curves[fm]['est'][:, :len(w_common)]
        est = est_all[np.searchsorted(P_SET, good)]
        I_est = i_omega(est_all)                                  # generator I: every P, as generated

        for k, p in enumerate(good):
            rows_M.append(pd.DataFrame({'FM': fm, 'P': p, 'W': w_common,
                                        'M_mean': m_mean[k], 'M_err': m_err[k]}))
        rows_I.append(pd.DataFrame({'FM': fm, 'W': w_common, 'I_mean': I_mean,
                                    'I_err': I_err, 'I_Est_M': I_est}))

        ax = axes[0, c]
        for kk, p in enumerate(P_SET):
            _, _, M, w_real, unstable = per_p[p]
            r_mean, r_err = M.mean(axis=0), jk_error(M, axis=0)
            if unstable:
                ax.plot([], [], 'o', color=colors[kk], label=f'P={p} (rollout unstable)')
                ax.errorbar(w_real, r_mean, r_err, fmt='o', ms=4, color=colors[kk], capsize=2)
                continue
            k = good.index(p)
            ax.fill_between(w_common, m_mean[k] - m_err[k], m_mean[k] + m_err[k],
                            color=colors[kk], alpha=0.12, lw=0)
            ax.plot(w_common, est[k], ':', color=colors[kk], lw=1, alpha=0.7)
            ax.plot(w_common, m_mean[k], '-', color=colors[kk], lw=2, label=f'P={p}')
            k = kk
            ax.errorbar(w_real[:N_SEED], r_mean[:N_SEED], r_err[:N_SEED], fmt='o', ms=4,
                        color=colors[k], capsize=2)
            if len(w_real) > N_SEED:
                ax.errorbar(w_real[N_SEED:], r_mean[N_SEED:], r_err[N_SEED:], fmt='s', ms=6,
                            mfc='white', color=colors[k], capsize=2)
        ylim_from(ax, m_mean, est)
        ax.set_title(f'FM={fm} (a = {FM_TO_A[fm]} fm): M($\\omega$)')
        ax.set_xlabel(r'$\omega$')
        ax.legend(title='line: DNN, dotted: Est_M\n●: real seed, □: real check',
                  fontsize=9, title_fontsize=8)

        ax = axes[1, c]
        ax.fill_between(w_common, I_mean - I_err, I_mean + I_err, color='#2a78d6', alpha=0.2, lw=0)
        ax.plot(w_common, I_est, ':', color='0.3', lw=1.2, label='from Est_M (generator fit)')
        ax.plot(w_common, I_mean, '-', color='#2a78d6', lw=2, label='DNN from real seeds' + MODE_LABEL[args.I_mode])
        if args.I_mode == 'signed':
            ax.axhline(0, color='0.5', lw=0.8)
        ylim_from(ax, I_mean, I_est)
        used = [p for p in good if p >= MIN_P]
        ax.set_title(f'FM={fm}: {"|I|" if args.I_mode == "abs" else "I"}($\\omega$), pairs from $p_z$ = '
                     f'{", ".join(map(str, used))}')
        ax.set_xlabel(r'$\omega$')
        ax.legend(fontsize=9)

        for p in good:                            # real check points vs the DNN mean curve
            pred, M, w_real = per_p[p][1], per_p[p][2], per_p[p][3]
            for z in range(N_SEED, len(w_real)):
                print(f'  FM={fm} P={p} Z={z} omega={w_real[z]:.3f}: real {M[:, z].mean():.4f} '
                      f'+- {jk_error(M[:, z]):.4f}   DNN {pred[:, z].mean():.4f} +- {jk_error(pred[:, z]):.4f}')

    axes[0, 0].set_ylabel('M')
    axes[1, 0].set_ylabel('|I|' if args.I_mode == 'abs' else 'I')
    fig.suptitle(f'Real lattice data fed to DeepQCD {args.model} ({os.path.basename(path)}): '
                 f'seeds = first 4 real points (Z = 0..3) per P, bands = jackknife error'
                 + MODE_TITLE[args.I_mode])
    fig.tight_layout()
    tag = args.model + ('' if args.I_mode == 'clip' else f'_{args.I_mode}I')
    fig.savefig(os.path.join(IMG_DIR, f'real_M_I_{tag}.png'), dpi=PLOT_DPI)
    pd.concat(rows_M).to_csv(os.path.join(OUT_DIR, f'real_M_all_P_{args.model}.csv'), index=False)
    pd.concat(rows_I).to_csv(os.path.join(OUT_DIR, f'real_I_omega_{tag}.csv'), index=False)
    print(f'wrote Images/real_M_I_{tag}.png, real_M_all_P_{args.model}.csv, '
          f'real_I_omega_{tag}.csv to {OUT_DIR}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='direct-noa-nolevel-wide', help='model spec of the checkpoint')
    ap.add_argument('--test_fm', default='', help='use that leave-out checkpoint instead of the all-FM one')
    ap.add_argument('--I_mode', default='clip', choices=I_MODES,
                    help='clip: test_error.py rule (negative pairs dropped, 0 if all negative); '
                         'abs: mean |pair estimate|; signed: mean pair estimate, negatives kept')
    ap.add_argument('--device', default='auto')
    main(ap.parse_args())
