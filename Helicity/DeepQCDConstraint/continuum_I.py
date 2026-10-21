"""Continuum limit (a -> 0) of the DeepQCDConstraint I(omega), same method as
Code/continuum_I_jackknife.py.

At every omega the three spacings (a = 0.0888 / 0.1207 / 0.151 fm) are fit with
    I(omega, a) = I0(omega) + c(omega) a^2      (main result)
    I(omega, a) = I0(omega) + c(omega) a        (systematic check)
The fit is linear in the per-FM I, I0 = sum_FM w_FM I_FM, and the ensembles are
independent, so sigma_stat^2 = sum_FM w_FM^2 sigma_JK,FM^2 exactly.
Systematic = |I0_a2 - I0_a|; sigma_tot = stat (+) sys in quadrature.  The reference
curve (Est_M truth for the rollout, the generator I for real data) is extrapolated the
same way.

Inputs (../Data/DeepQCDConstraint/)
    real       real_I_omega_{tag}.csv            (plot_real_M_I.py)
    rollout    rollout_I_omega_{layout}_dw{dw}.csv  (evaluate.py)
Writes
    I_omega_continuum_{source}.csv   W, I0, sigma_stat, sigma_sys, sigma_tot, I0_lin_a, c2,
                                     I0_ref, I_FM<fm>, sigma_FM<fm>
    images/I_omega_continuum_{source}.png   (drawn by plot.py)

Run from Helicity/DeepQCDConstraint:
    python continuum_I.py                                   # both, default files
    python continuum_I.py --real real_I_omega_dqc_all_dw0.2.csv --rollout rollout_I_omega_lattice_dw0.5.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

from common import FM_TO_A, OUT_DIR
from plot import plot_continuum


def weights(a, power):
    """Row 0 of the least-squares operator of I = I0 + c a^power: I0 = w @ I_FM."""
    X = np.column_stack([np.ones_like(a), a ** power])
    return np.linalg.pinv(X)


def continuum(df, val, err, ref):
    """Pointwise a^2 and a fits over the FMs of ``df`` (FM, W, val, err, ref)."""
    fms = sorted(df.FM.unique())
    a = np.array([FM_TO_A[f] for f in fms])
    piv = {c: df.pivot(index='W', columns='FM', values=c)[fms].dropna() for c in (val, err, ref)}
    W = piv[val].index.intersection(piv[err].index).intersection(piv[ref].index)
    I, S, R = (piv[c].loc[W].to_numpy().T for c in (val, err, ref))      # (n_FM, n_W)
    op2, op1 = weights(a, 2), weights(a, 1)
    I0, c2 = op2 @ I
    I0_lin = (op1 @ I)[0]
    stat = np.sqrt((op2[0][:, None] ** 2 * S ** 2).sum(axis=0))
    sys = np.abs(I0 - I0_lin)
    out = pd.DataFrame({'W': W, 'I0': I0, 'sigma_stat': stat, 'sigma_sys': sys,
                        'sigma_tot': np.hypot(stat, sys), 'I0_lin_a': I0_lin, 'c2': c2,
                        'I0_ref': (op2 @ R)[0]})
    for k, f in enumerate(fms):
        out[f'I_FM{f}'], out[f'sigma_FM{f}'] = I[k], S[k]
    return out, fms


def main(args):
    for source, fname, ref in [('real', args.real, 'I_Est_M'), ('rollout', args.rollout, 'I_true')]:
        path = os.path.join(OUT_DIR, fname)
        if not os.path.exists(path):
            print(f'{source}: {path} not found, skipped')
            continue
        out, fms = continuum(pd.read_csv(path), 'I_mean', 'I_err', ref)
        out.to_csv(os.path.join(OUT_DIR, f'I_omega_continuum_{source}.csv'), index=False)
        plot_continuum(source, fname)
        pk = out.I0.idxmax()
        print(f'{source}: I0 peak {out.I0[pk]:.4f} +- {out.sigma_tot[pk]:.4f} at omega {out.W[pk]:.1f} '
              f'(reference {out.I0_ref.max():.4f}); RMS(I0 - reference) {np.sqrt(((out.I0 - out.I0_ref) ** 2).mean()):.4f}')
    print(f'wrote I_omega_continuum_*.csv and images/I_omega_continuum_*.png to {OUT_DIR}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', default='real_I_omega_dqc_all_dw0.2.csv')
    ap.add_argument('--rollout', default='rollout_I_omega_lattice_dw0.5.csv')
    main(ap.parse_args())
