"""Reconstruct/extrapolate the real-data M(omega) curves with the trained models.

For every lattice ensemble (FM) and every (P, Exp) replica:

  * seed a 4-point (w, m) lag window at the low-omega end of the synthetic
    omega grid, taking the *real* measured M wherever the real omega grid has
    that point and the synthetic mean elsewhere;
  * roll the window forward one synthetic-grid step (0.1) at a time.  At each
    step the per-P GB / RF / XGB model predicts M at the next omega and that
    prediction is fed back in as the newest lag -- real data is **never**
    consulted or substituted here, only within the initial seed window.

So only the first ``N_LAGS`` (seed) points can match the real data (and only
where the seed omega happens to fall on the real grid); every point at or
beyond the 5th omega step is a genuine, purely autoregressive model
extrapolation with no teacher forcing.  RF is the least reliable in that
tail -- a random forest cannot extrapolate past its training targets and
tends to flat-line under long autoregression.

The per-P model variant is chosen with ``--exp_id`` (default ``No`` -> the
``*_exp_No`` models, no Exp feature; ``Yes`` -> ``*_exp_Yes``).  The output
file name carries the variant so both can coexist:
    --exp_id No   -> Real_data_projected_{fm}_no_exp_v2.csv
    --exp_id Yes  -> Real_data_projected_{fm}_with_exp_v2.csv
Per train_error_summary.csv the two are within noise; Exp is an arbitrary
replica index, so for ``No`` it is simply not fed through the roll.

Run from ``Helicity/Code`` (paths are relative):
    python reconstruct_real_data.py                 # no_exp
    python reconstruct_real_data.py --exp_id Yes    # with_exp
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

pd.options.mode.chained_assignment = None

# Feature order the models were trained on (prepare_train_test_data.py):
# P [, Exp], then 4 lag (w, m) pairs, then w5.  The *_exp_Yes models carry the
# extra Exp column; *_exp_No models do not.
_LAGS = ['w1', 'm1', 'w2', 'm2', 'w3', 'm3', 'w4', 'm4', 'w5']
FEATURES = {'No': ['P'] + _LAGS, 'Yes': ['P', 'Exp'] + _LAGS}
MODEL_NAMES = ['GB', 'RF', 'XGB']
N_LAGS = 4          # number of (w, m) lag points feeding each prediction
W_DECIMALS = 1      # omega grid is compared after rounding to this many decimals

# exp_id -> output-filename token
EXP_TOKEN = {'No': 'no_exp', 'Yes': 'with_exp'}


def _wkey(w):
    """Integer key (tenths of omega) for exact dict lookups on the 0.1 grid."""
    return np.round(np.asarray(w, dtype=float) * 10).astype(int)


class Reconstruct_real_data:
    def __init__(self, data_dir='../Data', model_dir='../model', exp_id='No'):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.exp_id = exp_id
        self.features = FEATURES[exp_id]
        self.use_exp = exp_id == 'Yes'

    def _load_models(self, fm, ordered_ps):
        """{model_name: {P: estimator}} for one FM.

        model.py appends one estimator per P in ``sorted(P.unique())`` order,
        so element i corresponds to ``ordered_ps[i]`` -- NOT to ``P == i + 1``.
        """
        models = {}
        for name in MODEL_NAMES:
            path = os.path.join(self.model_dir, str(fm), f'{name}_exp_{self.exp_id}.sav')
            est_list = joblib.load(path)
            if len(est_list) != len(ordered_ps):
                raise ValueError(
                    f'FM={fm} {name}: {len(est_list)} per-P models but '
                    f'{len(ordered_ps)} P values {ordered_ps} in the data')
            models[name] = dict(zip(ordered_ps, est_list))
        return models

    def _load_synthetic(self, fm):
        """Synthetic replicas for one FM as columns [P, Exp, W, M], omega rounded."""
        per_fm = os.path.join(self.data_dir, f'generated_Synthetic_exp_data_{fm}.csv')
        if os.path.exists(per_fm):
            df = pd.read_csv(per_fm, usecols=['P', 'Exp', 'W', 'M'])
        else:
            df = pd.read_csv(os.path.join(self.data_dir, 'generated_Synthetic_exp_data.csv'),
                             usecols=['FM', 'P', 'Exp', 'W', 'M'])
            df = df[df.FM == fm][['P', 'Exp', 'W', 'M']].copy()
        df['W'] = np.round(df['W'], W_DECIMALS)
        return df

    @staticmethod
    def _real_lut(real_piv, exps):
        """{wkey: (n_exp,) real M aligned to ``exps``, NaN where that Exp lacks
        the point}.  ``real_piv`` is Exp x W (omega already rounded)."""
        lut = {}
        if real_piv is not None and real_piv.shape[1] > 0:
            aligned = real_piv.reindex(index=exps)
            for c in aligned.columns:
                lut[int(_wkey(c))] = aligned[c].to_numpy(dtype=float)
        return lut

    @staticmethod
    def _interleave(ws, ms):
        """row-wise [w1..w4] + [m1..m4] -> [w1, m1, w2, m2, w3, m3, w4, m4]."""
        out = np.empty((ws.shape[0], 2 * ws.shape[1]), dtype=float)
        out[:, 0::2] = ws
        out[:, 1::2] = ms
        return out

    def reconstruct_data(self):
        df_real_all = pd.read_csv(os.path.join(self.data_dir, 'All_p_w_m.csv'))
        df_real_all['W'] = np.round(df_real_all['W'], W_DECIMALS)

        for fm in sorted(df_real_all.FM.unique()):
            df_syn = self._load_synthetic(fm)
            df_real = df_real_all[df_real_all.FM == fm]

            ordered_ps = sorted(df_syn.P.unique())
            models = self._load_models(fm, ordered_ps)
            all_ws = np.array(sorted(df_syn.W.unique()), dtype=float)

            P_parts, E_parts, W_parts = [], [], []
            anchored_parts = []
            pred_parts = {name: [] for name in MODEL_NAMES}

            for p in ordered_ps:
                gen_piv = (df_syn[df_syn.P == p]
                           .pivot_table(index='Exp', columns='W', values='M', aggfunc='mean')
                           .sort_index())
                gen_piv = gen_piv.reindex(columns=sorted(gen_piv.columns))
                gen_ws = np.array(gen_piv.columns, dtype=float)

                if len(gen_ws) <= N_LAGS:
                    print(f'FM={fm} P={p}: only {len(gen_ws)} omega points (<= {N_LAGS}), skipped')
                    continue

                exps = gen_piv.index.to_numpy()
                n_exp = len(exps)
                seed_ws = gen_ws[:N_LAGS]

                real_p = df_real[df_real.P == p]
                real_piv = None
                if not real_p.empty:
                    real_piv = real_p.pivot_table(index='Exp', columns='W', values='M',
                                                  aggfunc='mean')
                real_lut = self._real_lut(real_piv, exps)

                # seed M: synthetic mean, with the real measurement substituted
                # at any seed omega the real grid covers
                seed_M = np.array(gen_piv.iloc[:, :N_LAGS].to_numpy(dtype=float), copy=True)
                for j, w in enumerate(seed_ws):
                    anchor = real_lut.get(int(_wkey(w)))
                    if anchor is not None:
                        have = np.isfinite(anchor)
                        seed_M[have, j] = anchor[have]
                if not np.isfinite(seed_M).all():
                    seed_M = (pd.DataFrame(seed_M).ffill(axis=1).bfill(axis=1)
                              .fillna(0.0).to_numpy())

                p_col = np.full(n_exp, p, dtype=float)
                exp_col = exps.astype(float)

                # seed rows: the (possibly real) seed value goes into every column
                for j, w in enumerate(seed_ws):
                    a = real_lut.get(int(_wkey(w)))
                    P_parts.append(p_col.copy())
                    E_parts.append(exps)
                    W_parts.append(np.full(n_exp, w))
                    anchored_parts.append(np.isfinite(a) if a is not None
                                          else np.zeros(n_exp, dtype=bool))
                    for name in MODEL_NAMES:
                        pred_parts[name].append(seed_M[:, j].copy())

                # per-model rolling lag window [w1, m1, ... w4, m4]
                lag_ws = np.tile(seed_ws, (n_exp, 1))
                state = {name: self._interleave(lag_ws, seed_M.copy()) for name in MODEL_NAMES}

                for w in all_ws[all_ws > seed_ws[-1]]:
                    w5 = np.full(n_exp, w, dtype=float)

                    step_val = {}
                    for name in MODEL_NAMES:
                        cols = ([p_col, exp_col, state[name], w5] if self.use_exp
                                else [p_col, state[name], w5])
                        X = pd.DataFrame(np.column_stack(cols), columns=self.features)
                        pred = np.asarray(models[name][p].predict(X), dtype=float)
                        step_val[name] = pred

                    P_parts.append(p_col.copy())
                    E_parts.append(exps)
                    W_parts.append(w5.copy())
                    anchored_parts.append(np.zeros(n_exp, dtype=bool))
                    for name in MODEL_NAMES:
                        pred_parts[name].append(step_val[name])
                        # slide: drop (w1, m1), append (w5, value used)
                        state[name] = np.column_stack(
                            [state[name][:, 2:], w5, step_val[name]])

                n_extrap = int((all_ws > seed_ws[-1]).sum())
                print(f'FM={fm} P={p} done  (seed anchored omega<= {seed_ws[-1]:.1f}, '
                      f'{n_extrap} pure-autoregressive extrapolated steps)')

            if not P_parts:
                print(f'FM={fm}: no usable (P, Exp) groups, nothing saved')
                continue

            df_out = pd.DataFrame({
                'FM': fm,
                'P': np.concatenate(P_parts).astype(int),
                'Exp': np.concatenate(E_parts).astype(int),
                'W': np.round(np.concatenate(W_parts), W_DECIMALS),
                'anchored': np.concatenate(anchored_parts).astype(int),
                'M_GB': np.concatenate(pred_parts['GB']),
                'M_RF': np.concatenate(pred_parts['RF']),
                'M_XGB': np.concatenate(pred_parts['XGB']),
            }).sort_values(['P', 'Exp', 'W']).reset_index(drop=True)

            token = EXP_TOKEN[self.exp_id]
            out_path = os.path.join(self.data_dir,
                                    f'Real_data_projected_{fm}_{token}_v2.csv')
            df_out.to_csv(out_path, index=False)
            print(f'FM={fm} saved -> {out_path}  ({len(df_out)} rows)')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='../Data/2026',
                        help="Directory with All_p_w_m.csv and generated_Synthetic_exp_data*.csv")
    parser.add_argument("--model_dir", default='../model/2026',
                        help="Directory with {model_dir}/{fm}/{GB,RF,XGB}_exp_{id}.sav")
    parser.add_argument("--exp_id", default='No', choices=['No', 'Yes'],
                        help="Which per-P model variant to roll with (default: No / without_exp)")
    args = parser.parse_args()

    rd = Reconstruct_real_data(data_dir=args.data_dir, model_dir=args.model_dir,
                               exp_id=args.exp_id)
    rd.reconstruct_data()
