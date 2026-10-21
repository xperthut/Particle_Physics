"""Every DeepQCDConstraint image, drawn from the CSVs in ../Data/DeepQCDConstraint/ only
(no model, no generator file).  train.py, evaluate.py, plot_real_M_I.py and
continuum_I.py call these functions after writing their CSVs; run this script to
redraw everything without rerunning them.

    image                                       from
    images/train_loss_{tag}.png                 train_history_{tag}.csv
    images/rollout_{layout}_dw{dw}_FM{fm}.png   rollout_M_all_P_*.csv, rollout_seeds_*.csv, rollout_I_omega_*.csv
    images/rollout_{layout}_dw{dw}_I_all_FM.png rollout_I_omega_{layout}_dw{dw}.csv
    images/real_M_I_{tag}_FM{fm}.png            real_M_all_P_{tag}.csv, real_vs_dnn_{tag}.csv, real_I_omega_{tag}.csv
    images/real_I_all_FM_{tag}.png              real_I_omega_{tag}.csv
    images/I_omega_continuum_{source}.png       I_omega_continuum_{source}.csv

Run from Helicity/DeepQCDConstraint:  python plot.py [--only train,rollout,real,continuum]
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FM_TO_A, IMG_DIR, OUT_DIR, PLOT_DPI

OLD_LABEL = 'DeepQCD direct-noa-nolevel (signed pair mean)'
SEEDS = {'grid': 'omega = 0, dw, 2dw, 3dw for every P', 'lattice': 'real lattice omegas P z d1, z = 0..3'}


def _colors():
    return plt.rcParams['axes.prop_cycle'].by_key()['color']


def _csv(name):
    return pd.read_csv(os.path.join(OUT_DIR, name))


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, name), dpi=PLOT_DPI)
    plt.close(fig)
    return name


def plot_train_loss(tag):
    hist = _csv(f'train_history_{tag}.csv')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(hist.epoch, hist.train_loss, label='train')
    ax.semilogy(hist.epoch, hist.val_loss, label='val (held-out replicas)')
    ax.set_xlabel('epoch')
    ax.set_ylabel('MSE (M / per-P seed scale)')
    ax.set_title(f'DeepQCDConstraint {tag}')
    ax.legend()
    return [_save(fig, f'train_loss_{tag}.png')]


def _M_panel(ax, gm, colors):
    for k, (p, g) in enumerate(gm.groupby('P')):
        ax.fill_between(g.W, g.M_mean - g.M_err, g.M_mean + g.M_err, color=colors[k], alpha=0.15, lw=0)
        ax.plot(g.W, g.Est_M, ':', color=colors[k], lw=1.2)
        ax.plot(g.W, g.M_mean, '-', color=colors[k], lw=2, label=f'P={p}')
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel('M')


def plot_rollout(layout, dw):
    key = f'{layout}_dw{dw}'
    dm, ds, di = _csv(f'rollout_M_all_P_{key}.csv'), _csv(f'rollout_seeds_{key}.csv'), _csv(f'rollout_I_omega_{key}.csv')
    colors, out = _colors(), []
    for fm in sorted(dm.FM.unique()):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax = axes[0]
        _M_panel(ax, dm[dm.FM == fm], colors)
        for k, (p, g) in enumerate(ds[ds.FM == fm].groupby('P')):
            ax.plot(g.W, g.M_mean, 'o', ms=4, color=colors[k])
        ax.set_title(f'held-out FM={fm} (a = {FM_TO_A[fm]} fm): M($\\omega$)')
        ax.legend(title='line: constrained DNN, dotted: Est_M, ●: seeds', fontsize=9, title_fontsize=8)
        ax = axes[1]
        g = di[di.FM == fm]
        ax.fill_between(g.W, g.I_mean - g.I_err, g.I_mean + g.I_err, color='#2a78d6', alpha=0.2, lw=0)
        ax.plot(g.W, g.I_true, ':', color='0.2', lw=1.5, label='Est_M (truth)')
        ax.plot(g.W, g.I_mean, '-', color='#2a78d6', lw=2, label='DeepQCDConstraint')
        if 'I_DeepQCD' in g:
            ax.plot(g.W, g.I_DeepQCD, '--', color='#d65a2a', lw=1.3, label=OLD_LABEL)
        ax.axhline(0, color='0.6', lw=0.8)
        ax.set_title(f'FM={fm}: I($\\omega$)')
        ax.set_xlabel(r'$\omega$')
        ax.set_ylabel('I')
        ax.legend(fontsize=9)
        fig.suptitle(f'Held-out FM={fm}, {layout} seeds ({SEEDS[layout]}), rollout dw = {dw}; '
                     f'bands = jackknife error over replicas', fontsize=10)
        out.append(_save(fig, f'rollout_{key}_FM{fm}.png'))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for c, (fm, g) in enumerate(di.groupby('FM')):
        ax.fill_between(g.W, g.I_mean - g.I_err, g.I_mean + g.I_err, color=colors[c], alpha=0.15, lw=0)
        ax.plot(g.W, g.I_mean, '-', color=colors[c], lw=2, label=f'FM={fm} (a = {FM_TO_A[fm]} fm)')
        ax.plot(g.W, g.I_true, ':', color=colors[c], lw=1.5)
        if 'I_DeepQCD' in g:
            ax.plot(g.W, g.I_DeepQCD, '--', color=colors[c], lw=1, alpha=0.8)
    ax.axhline(0, color='0.6', lw=0.8)
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel('I')
    ax.set_title(f'I($\\omega$), held-out FMs, {layout} seeds, rollout dw = {dw}', fontsize=11)
    ax.legend(title=f'solid: DeepQCDConstraint ± JK, dotted: Est_M (truth),\ndashed: {OLD_LABEL}',
              fontsize=9, title_fontsize=8)
    out.append(_save(fig, f'rollout_{key}_I_all_FM.png'))
    return out


def plot_real(tag, xmax=20.0):
    dm, di, dc = _csv(f'real_M_all_P_{tag}.csv'), _csv(f'real_I_omega_{tag}.csv'), _csv(f'real_vs_dnn_{tag}.csv')
    model, dw = tag.rsplit('_dw', 1)
    colors, out = _colors(), []
    for fm in sorted(dm.FM.unique()):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax = axes[0]
        gm = dm[(dm.FM == fm) & (dm.W <= xmax)]
        _M_panel(ax, gm, colors)
        gc = dc[dc.FM == fm]
        for k, (p, g) in enumerate(gc.groupby('P')):
            s, c = g[g.seed], g[~g.seed]
            ax.errorbar(np.r_[0.0, s.W], np.r_[0.0, s.real_mean], np.r_[0.0, s.real_err], fmt='o', ms=4,
                        color=colors[k], capsize=2)             # Z = 0 is the M = 0 anchor
            if len(c):
                ax.errorbar(c.W, c.real_mean, c.real_err, fmt='s', ms=6, mfc='white', color=colors[k], capsize=2)
        hi = max(gm.M_mean.max(), gc.real_mean.max())
        ax.set_xlim(-0.3, xmax)
        ax.set_ylim(-0.05 * hi, 1.15 * hi)
        ax.set_title(f'FM={fm} (a = {FM_TO_A[fm]} fm): M($\\omega$)')
        ax.legend(title='line: constrained DNN, dotted: Est_M\n●: real seed, □: real check',
                  fontsize=9, title_fontsize=8)
        ax = axes[1]
        g = di[di.FM == fm]
        ax.fill_between(g.W, g.I_mean - g.I_err, g.I_mean + g.I_err, color='#2a78d6', alpha=0.2, lw=0)
        ax.plot(g.W, g.I_Est_M, ':', color='0.3', lw=1.2, label='from Est_M (generator fit)')
        ax.plot(g.W, g.I_mean, '-', color='#2a78d6', lw=2, label='constrained DNN from real seeds')
        ax.axhline(0, color='0.6', lw=0.8)
        ax.set_xlim(-0.3, xmax)
        ax.set_title(f'FM={fm}: I($\\omega$) (every pair gives the same I)')
        ax.set_xlabel(r'$\omega$')
        ax.set_ylabel('I')
        ax.legend(fontsize=9)
        fig.suptitle(f'FM={fm}: real lattice data fed to DeepQCDConstraint ({model}.pt), seeds = Z 0..3 per P, '
                     f'rollout dw = {dw}, bands = jackknife error', fontsize=10)
        out.append(_save(fig, f'real_M_I_{tag}_FM{fm}.png'))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for c, (fm, g) in enumerate(di.groupby('FM')):
        ax.fill_between(g.W, g.I_mean - g.I_err, g.I_mean + g.I_err, color=colors[c], alpha=0.15, lw=0)
        ax.plot(g.W, g.I_mean, '-', color=colors[c], lw=2, label=f'FM={fm} (a = {FM_TO_A[fm]} fm)')
        ax.plot(g.W, g.I_Est_M, ':', color=colors[c], lw=1.5)
    ax.axhline(0, color='0.6', lw=0.8)
    ax.set_xlim(-0.3, xmax)
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel('I')
    ax.set_title(f'I($\\omega$) from real lattice seeds, DeepQCDConstraint ({model}.pt), dw = {dw}', fontsize=11)
    ax.legend(title='solid: constrained DNN ± JK, dotted: from Est_M (generator fit)', fontsize=9, title_fontsize=8)
    out.append(_save(fig, f'real_I_all_FM_{tag}.png'))
    return out


CONTINUUM = {'real': ('Continuum I($\\omega$) from real lattice seeds', 'the generator I (Est_M)'),
             'rollout': ('Continuum I($\\omega$), held-out-FM synthetic rollout', 'Est_M (truth)')}


def plot_continuum(source, subtitle=''):
    out = _csv(f'I_omega_continuum_{source}.csv')
    title, ref_label = CONTINUUM[source]
    fms = sorted(int(c[4:]) for c in out.columns if re.fullmatch(r'I_FM\d+', c))
    colors = _colors()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    w = out.W
    for k, f in enumerate(fms):
        ax.plot(w, out[f'I_FM{f}'], '-', color=colors[k], lw=1.2, alpha=0.7, label=f'FM={f} (a = {FM_TO_A[f]} fm)')
    ax.fill_between(w, out.I0 - out.sigma_tot, out.I0 + out.sigma_tot, color='k', alpha=0.12, lw=0,
                    label='a → 0: stat ⊕ sys')
    ax.fill_between(w, out.I0 - out.sigma_stat, out.I0 + out.sigma_stat, color='k', alpha=0.25, lw=0,
                    label='a → 0: stat (jackknife)')
    ax.plot(w, out.I0, '-', color='k', lw=2.2, label=r'a → 0, $I_0 + c\,a^2$')
    ax.plot(w, out.I0_lin_a, '--', color='k', lw=1, label=r'a → 0, $I_0 + c\,a$ (systematic)')
    ax.plot(w, out.I0_ref, ':', color='#d65a2a', lw=1.8, label=f'a → 0 of {ref_label}')
    ax.axhline(0, color='0.6', lw=0.8)
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel(r'$I(\omega)$')
    ax.set_title(title + (f' ({subtitle})' if subtitle else ''), fontsize=11)
    ax.legend(fontsize=8.5)
    return [_save(fig, f'I_omega_continuum_{source}.png')]


def _tags(pattern):
    """The {} part of every OUT_DIR file matching pattern 'prefix_{}.csv'."""
    pre, post = pattern.split('{}')
    return sorted(os.path.basename(p)[len(pre):-len(post)] for p in glob.glob(os.path.join(OUT_DIR, pre + '*' + post)))


def main(args):
    only = set(args.only.split(','))
    made = []
    if 'train' in only:
        for tag in _tags('train_history_{}.csv'):
            made += plot_train_loss(tag)
    if 'rollout' in only:
        for key in _tags('rollout_M_all_P_{}.csv'):
            layout, dw = key.split('_dw')
            made += plot_rollout(layout, dw)
    if 'real' in only:
        for tag in _tags('real_M_all_P_{}.csv'):
            made += plot_real(tag, args.xmax)
    if 'continuum' in only:
        for source in _tags('I_omega_continuum_{}.csv'):
            if source in CONTINUUM:
                made += plot_continuum(source)
    print(f'wrote {len(made)} images to {IMG_DIR}:')
    for m in made:
        print('   ', m)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='train,rollout,real,continuum', help='comma list of image groups')
    ap.add_argument('--xmax', type=float, default=20.0, help='omega range of the real-data plots')
    main(ap.parse_args())
