"""Fill the missing 0.1-grid omega values of the real M(omega) data.

The real data sit on an irregular omega grid (omega = P * z * const, e.g.
0.196, 0.393, ... for FM=9 P=1).  For every (FM, P) this fills each 0.1-grid
omega in (0, max real omega] -- never beyond the measured range -- per Exp:

  1. fit  M(omega) = A * omega * exp(-s * omega)  to that (FM, P)'s mean real M
     (2 parameters per (FM, P); the same omega*exp(-s*omega) shape as
     synthetic_data_generator.generate_I_omega, but fit per P since the shared
     P**(-gamma) form misfits the real P dependence by 15-120%);
  2. for each Exp, take its residuals  M_exp - fit  at the real omegas (0 at the
     omega=0 anchor) and interpolate them linearly onto the grid;
  3. filled M = fit + interpolated residual.

So the fit supplies the shape between real points, and each filled curve
passes exactly through that Exp's own real values -- the real points are kept
unchanged and the Exp spread / cross-omega correlation carries over.

Output ``Data/2026/Real_data_filled.csv`` (FM, P, Exp, W, M, is_real): the
real rows at their exact omega (is_real=1) plus the filled 0.1-grid rows
(is_real=0).  Also ``Real_data_fill_params.csv`` (FM, P, A, s, max_rel_err of
the fit to the mean).

Run from ``Helicity/Code``:
    python fill_real_data.py
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

GRID_STEP = 0.1


def shape(w, A, s):
    return A * w * np.exp(-s * w)


def fill_group(t, grid_step=GRID_STEP):
    """Real rows of one (FM, P) -> (filled long frame, fit params dict)."""
    piv = t.pivot_table(index='Exp', columns='W', values='M', aggfunc='mean').sort_index(axis=1)
    piv = piv.drop(columns=[c for c in piv.columns if c == 0.0])     # anchor re-added below
    w_real = piv.columns.to_numpy(dtype=float)
    mean_M = piv.mean(axis=0).to_numpy()

    (A, s), _ = curve_fit(shape, w_real, mean_M, p0=[mean_M[0] / w_real[0], 0.1], maxfev=20000)
    max_rel = float(np.max(np.abs(shape(w_real, A, s) - mean_M) / np.abs(mean_M)))

    grid = np.round(np.arange(grid_step, w_real.max() + 1e-9, grid_step), 10)
    grid = grid[~np.isclose(grid[:, None], w_real[None, :]).any(axis=1)]   # only missing omegas

    w_knots = np.r_[0.0, w_real]
    resid = piv.to_numpy(dtype=float) - shape(w_real, A, s)
    fit_grid = shape(grid, A, s)
    filled = np.empty((len(piv), len(grid)))
    for i, r in enumerate(resid):
        ok = np.r_[True, np.isfinite(r)]
        filled[i] = fit_grid + np.interp(grid, w_knots[ok], np.r_[0.0, r][ok])

    exps = piv.index.to_numpy()
    out = pd.DataFrame({
        'Exp': np.repeat(exps, len(grid)),
        'W': np.tile(grid, len(exps)),
        'M': filled.ravel(),
        'is_real': 0,
    })
    real = t[['Exp', 'W', 'M']].assign(is_real=1)
    return pd.concat([real, out], ignore_index=True), {'A': A, 's': s, 'max_rel_err': max_rel}


def main(data_dir):
    real = pd.read_csv(os.path.join(data_dir, 'All_p_w_m.csv'))
    parts, params = [], []
    for (fm, p), t in real.groupby(['FM', 'P']):
        df, par = fill_group(t)
        parts.append(df.assign(FM=fm, P=p))
        params.append({'FM': fm, 'P': p, **par})
        n_fill = int((df.is_real == 0).sum() / max(t.Exp.nunique(), 1))
        print(f'FM={fm} P={p}: A={par["A"]:.4f} s={par["s"]:+.4f} '
              f'fit max|rel err|={par["max_rel_err"]:.3f}, {n_fill} omegas filled')

    out = (pd.concat(parts, ignore_index=True)[['FM', 'P', 'Exp', 'W', 'M', 'is_real']]
           .sort_values(['FM', 'P', 'Exp', 'W']).reset_index(drop=True))
    out_path = os.path.join(data_dir, 'Real_data_filled.csv')
    out.to_csv(out_path, index=False)
    pd.DataFrame(params).to_csv(os.path.join(data_dir, 'Real_data_fill_params.csv'), index=False)
    print(f'saved -> {out_path}  ({len(out)} rows)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../Data/2026', help='Directory with All_p_w_m.csv')
    main(parser.parse_args().data_dir)
