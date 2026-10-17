"""Redraw every figure in Data/2026/Images from the data already on disk.

Nothing is trained, projected or refit: each figure is drawn by the plotting
function of the pipeline script that made it, fed from the CSVs that script
wrote.  Before a figure is drawn every input file is checked; a figure with
missing inputs is skipped (the missing files are listed) and the rest still run.

Figures (all written to Data/2026/Images/):
    real_diagnostics   real_M_jackknife, mean_median_and_qq, Mean_vs_W_FM_*, Std_vs_W_FM_*
    real_vs_synthetic  M_real_vs_synthetic_FM_*[_full]
    I_estimated        I(w)_estimated
    model_selection    model_selection_FM_*
    M_curves           M_curves_<best form>
    projected_I        projected_I_no_exp_*            (top-3 models by test error)
    best_model_I       best_model_I_no_exp_<best>
    I_all_FM           I_omega_all_FM_<best>_no_exp
    continuum          I_omega_continuum.{png,pdf}
    continuum_jk       I_omega_continuum_jk_<model>[_anchored].{png,pdf}, one per CSV present
    continuum_M        I_omega_continuum_M_<model>[_anchored].{png,pdf}, one per CSV present

Three plotting functions also rewrite the small CSV of what they draw
(real_M_jackknife.csv, M_curves_<form>.csv, best_model_I_no_exp_<best>.csv);
they are recomputed from the same inputs, so their contents do not change.

Usage (works from any directory):
    python plot_all.py                          # check inputs, plot everything available
    python plot_all.py --check                  # only report which figures have their data
    python plot_all.py --only continuum_jk projected_I
"""

import argparse
import glob
import os
import re
import sys
import traceback
from dataclasses import dataclass
from typing import Callable, List

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)                   # the pipeline modules use paths relative to Helicity/Code

import matplotlib
matplotlib.use("Agg")            # script context: render to files, no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# synthetic_data_generator sets figure.figsize / font.size on import; keep that
# style for its own figures only, as when it is run on its own
with matplotlib.rc_context():
    import synthetic_data_generator as sdg
    SDG_RC = {k: matplotlib.rcParams[k] for k in ('figure.figsize', 'font.size')}
import combine_I_omega as cio
import continuum_extrapolation as ce
import continuum_I_jackknife as cij
import continuum_M_extrapolation as cme
import model_selection as ms
import test_error as te

DATA_DIR = '../Data/2026'
IMG_DIR = os.path.join(DATA_DIR, 'Images')
VARIANT = te.VARIANT
REAL = os.path.join(DATA_DIR, 'All_p_w_m.csv')
RANKING = os.path.join(DATA_DIR, f'test_error_ranking_{VARIANT}.csv')


def data(name):
    return os.path.join(DATA_DIR, name)


@dataclass
class Job:
    group: str
    name: str
    inputs: List[str]
    run: Callable[[], None]


# --------------------------------------------------------------------------- real data / synthetic

def jobs_synthetic():
    def real_diagnostics():
        df = pd.read_csv(REAL)
        with matplotlib.rc_context(SDG_RC):
            sdg.plot_real_jackknife(df)
            sdg.plot_mean_median_and_qq(df)
            sdg.plot_mean_vs_w(df)
            sdg.plot_std_vs_w(df)

    def real_vs_synthetic():
        df = pd.read_csv(REAL)
        new_M = pd.read_csv(data('new_M.csv'))
        mk_model = ms.best_model()
        with matplotlib.rc_context(SDG_RC):
            for fm in sorted(df.FM.unique()):
                est = new_M[new_M.FM == fm][['P', 'W', 'Est_M']]
                if est.empty:
                    raise ValueError(f'new_M.csv has no rows for FM={fm}')
                df_g = sdg.real_mean_M(df, fm)
                for full in (False, True):
                    sdg.plot_real_vs_synthetic(df_g, est, fm, mk_model, full_range=full)

    def I_estimated():
        with matplotlib.rc_context(SDG_RC):
            sdg.plot_I_omega(pd.read_csv(data('I_omega_estimated.csv')))

    return [
        Job('real_diagnostics', 'real_diagnostics', [REAL], real_diagnostics),
        Job('real_vs_synthetic', 'real_vs_synthetic',
            [REAL, data('new_M.csv'), data('model_selection_summary.csv')], real_vs_synthetic),
        Job('I_estimated', 'I_estimated', [data('I_omega_estimated.csv')], I_estimated),
    ]


# --------------------------------------------------------------------------- model selection

def jobs_model_selection():
    fits_csv = data('model_selection_fits.csv')

    def model_selection():
        df = ms.load_real_data()
        fits = pd.read_csv(fits_csv)
        for fm in sorted(df.FM.unique()):
            cells, _ = ms.load_cells(df, fm)
            z2_max = float(np.max((cells.W / cells.P) ** 2))
            f = fits[fits.FM == fm]
            theta = {r.model: [r[f'p_{pn}'] for pn in ms.MODELS[r.model][1]]
                     for _, r in f.iterrows()}
            # as in model_selection.main: best three by QAICc, plus M1 for reference
            top = f.sort_values('QAICc').model.head(3).tolist()
            if 'M1' not in top:
                top.append('M1')
            ms.plot_fm(fm, cells, theta, z2_max, top)

    return [
        Job('model_selection', 'model_selection', [REAL, fits_csv], model_selection),
        Job('M_curves', 'M_curves', [REAL, fits_csv, data('model_selection_summary.csv')],
            lambda: ms.plot_curves(ms.best_model())),
    ]


# --------------------------------------------------------------------------- projections (test_error.py)

def projected_fms():
    out = []
    for path in glob.glob(data(f'Projected_M_mean_err_{VARIANT}_*.csv')):
        m = re.search(rf'Projected_M_mean_err_{VARIANT}_(.+)\.csv$', path)
        out.append(m.group(1))
    return sorted(out, key=float)


def w_anchor(fm):
    """Largest real-anchored omega in the projected file (0 for the plain seed-only roll)."""
    path = data(f'Real_data_projected_{fm}_{VARIANT}_v2.csv')
    if not os.path.exists(path) or 'anchored' not in pd.read_csv(path, nrows=0).columns:
        return 0.0
    d = pd.read_csv(path, usecols=['W', 'anchored'])
    d = d[d.anchored == 1]
    return float(np.round(d.W, te.W_DECIMALS).max()) if not d.empty else 0.0


_projected_cache = {}


def load_projected():
    """(top model names, overall ranking indexed by model, per-FM tuples as in Test_Error.run)."""
    if not _projected_cache:
        rank = pd.read_csv(RANKING).sort_values('overall_rank')
        real = pd.read_csv(REAL, usecols=['FM', 'P', 'Exp', 'W', 'M'])
        per_fm = []
        for fm in projected_fms():
            df_ms = pd.read_csv(data(f'Projected_M_mean_err_{VARIANT}_{fm}.csv'))
            i_path = data(f'Projected_I_{VARIANT}_{fm}.csv')
            df_I = pd.read_csv(i_path) if os.path.exists(i_path) else None
            names = [n for n in te.MODEL_NAMES if f'Mean_M_{n}' in df_ms.columns]
            per_fm.append((fm, VARIANT, df_ms, df_I, w_anchor(fm), names,
                           real[real.FM.astype(str) == fm]))
        _projected_cache.update(top=rank.model.head(te.TOP_K).tolist(),
                                overall=rank.set_index('model'), per_fm=per_fm)
    c = _projected_cache
    return c['top'], c['overall'], c['per_fm']


def jobs_projected():
    fms = projected_fms()
    ms_files = [data(f'Projected_M_mean_err_{VARIANT}_{fm}.csv') for fm in fms]
    I_files = [data(f'Projected_I_{VARIANT}_{fm}.csv') for fm in fms]
    if not fms:                  # nothing found: report the expected pattern as missing
        ms_files = [data(f'Projected_M_mean_err_{VARIANT}_<fm>.csv')]
        I_files = [data(f'Projected_I_{VARIANT}_<fm>.csv')]
    tester = te.Test_Error(data_dir=DATA_DIR)

    def projected_I():
        top, overall, per_fm = load_projected()
        for fm, variant, df_ms, df_I, w_anc, names, real_fm in per_fm:
            plot_names = [n for n in top if n in names] or names
            tester._plot(fm, variant, df_ms, df_I, w_anc, plot_names, real_fm, overall)

    def best_model_I():
        top, overall, per_fm = load_projected()
        tester._plot_best(top[0], per_fm, overall)

    def I_all_FM():
        top, _, _ = load_projected()
        best = top[0]
        w_real = pd.read_csv(REAL, usecols=['FM', 'W']).groupby('FM').W.max().to_dict()
        cio.plot(cio.collect(DATA_DIR, best, VARIANT), w_real, best, VARIANT,
                 os.path.join(IMG_DIR, f'I_omega_all_FM_{best}_{VARIANT}.png'))
        print(f'plot -> {IMG_DIR}/I_omega_all_FM_{best}_{VARIANT}.png')

    return [
        Job('projected_I', 'projected_I', [REAL, RANKING] + ms_files + I_files, projected_I),
        Job('best_model_I', 'best_model_I', [REAL, RANKING] + ms_files + I_files, best_model_I),
        Job('I_all_FM', 'I_all_FM', [REAL, RANKING] + I_files, I_all_FM),
    ]


# --------------------------------------------------------------------------- continuum limit

def fm_columns(df):
    """Spacings present as I_FM<fm> columns, in increasing order."""
    return sorted(int(c[4:]) for c in df.columns if re.fullmatch(r'I_FM\d+', c))


def jobs_continuum():
    src = data('I_omega_continuum.csv')

    def continuum():
        df = pd.read_csv(src)
        cols = sorted((c for c in df.columns if re.fullmatch(r'I_a\d+\.\d+', c)),
                      key=lambda c: float(c[3:]))
        a = np.array([float(c[3:]) for c in cols])
        fms = np.rint(a / ce.FM_TO_A).astype(int)
        ce.plot(df.omega.to_numpy(), fms, a, df[cols].to_numpy(),
                df.I_a0_a2.to_numpy(), df.I_a0_a.to_numpy(),
                os.path.join(IMG_DIR, 'I_omega_continuum'))

    jobs = [Job('continuum', 'continuum', [src], continuum)]

    # jackknife continuum (continuum_I_jackknife.py): one figure per CSV present
    jk = sorted(glob.glob(data('I_omega_continuum_jk_*.csv')))
    for path in jk:
        m = re.search(r'I_omega_continuum_jk_([A-Za-z]+)(_anchored)?\.csv$', path)
        if not m:
            continue
        model, suffix = m.group(1), m.group(2) or ''

        def run(path=path, model=model, suffix=suffix):
            out = pd.read_csv(path)
            fms = fm_columns(out)
            a = np.array([cij.FM_TO_A[f] for f in fms])
            cij.plot(out, fms, a, model, 'anchored' if suffix else 'roll',
                     os.path.join(IMG_DIR, f'I_omega_continuum_jk_{model}{suffix}'))
        jobs.append(Job('continuum_jk', f'continuum_jk:{model}{suffix}', [path], run))
    if not jk:
        jobs.append(Job('continuum_jk', 'continuum_jk',
                        [data('I_omega_continuum_jk_<model>.csv')], lambda: None))

    # continuum on M (continuum_M_extrapolation.py): one figure per CSV present
    mc = sorted(glob.glob(data('I_omega_continuum_M_*.csv')))
    for path in mc:
        m = re.search(r'I_omega_continuum_M_([A-Za-z]+)(_anchored)?\.csv$', path)
        if not m:
            continue
        model, suffix = m.group(1), m.group(2) or ''
        fit_csv = data(f'M_continuum_fit_{model}{suffix}.csv')

        def run(path=path, model=model, suffix=suffix, fit_csv=fit_csv):
            out = pd.read_csv(path)
            w_form = pd.read_csv(fit_csv).groupby('form').weight.first().reindex(cme.FORMS)
            weighting = 'equal' if np.allclose(w_form, 1.0 / len(cme.FORMS)) else 'aic'
            _, reach = cme.lattice_units(DATA_DIR)
            fms = fm_columns(out)
            a = np.array([cme.FM_TO_A[f] for f in fms])
            band = (r'jackknife $\oplus$ fit residual'
                    + (r' $\oplus$ GP latent draws' if (out.sigma_model > 0).any() else '')
                    + r' $\oplus$ cutoff-form spread')
            cme.plot(out, fms, a, w_form, weighting, max(reach.values()), model, band,
                     os.path.join(IMG_DIR, f'I_omega_continuum_M_{model}{suffix}'))
        jobs.append(Job('continuum_M', f'continuum_M:{model}{suffix}', [path, fit_csv, REAL], run))
    if not mc:
        jobs.append(Job('continuum_M', 'continuum_M',
                        [data('I_omega_continuum_M_<model>.csv')], lambda: None))
    return jobs


# --------------------------------------------------------------------------- driver

GROUPS = ['real_diagnostics', 'real_vs_synthetic', 'I_estimated', 'model_selection', 'M_curves',
          'projected_I', 'best_model_I', 'I_all_FM', 'continuum', 'continuum_jk', 'continuum_M']


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--check', action='store_true',
                        help='only check that every figure has its input data; plot nothing')
    parser.add_argument('--only', nargs='+', choices=GROUPS, metavar='GROUP',
                        help=f'plot only these groups: {", ".join(GROUPS)}')
    args = parser.parse_args()

    jobs = jobs_synthetic() + jobs_model_selection() + jobs_projected() + jobs_continuum()
    if args.only:
        jobs = [j for j in jobs if j.group in args.only]
    os.makedirs(IMG_DIR, exist_ok=True)

    results = []
    for job in jobs:
        missing = [p for p in job.inputs if not os.path.exists(p)]
        if missing:
            results.append((job.name, 'SKIP', 'missing: ' + ', '.join(missing)))
        elif args.check:
            results.append((job.name, 'READY', ''))
        else:
            print(f'\n=== {job.name}')
            try:
                job.run()
                results.append((job.name, 'OK', ''))
            except (Exception, SystemExit) as exc:
                traceback.print_exc()
                results.append((job.name, 'FAIL', f'{type(exc).__name__}: {exc}'))
            finally:
                plt.close('all')

    width = max(len(n) for n, *_ in results)
    print('\n' + '-' * 72)
    for name, status, note in results:
        print(f'{name:<{width}}  {status:<5}  {note}')
    counts = {s: sum(r[1] == s for r in results) for s in ('OK', 'READY', 'SKIP', 'FAIL')}
    print('-' * 72 + '\n' + ', '.join(f'{v} {k}' for k, v in counts.items() if v))
    return 1 if counts['FAIL'] else 0


if __name__ == '__main__':
    sys.exit(main())
