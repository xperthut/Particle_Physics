"""Report the training error (RMSE / R2) of the saved per-P models.

Mirrors train.py's layout: for every FM found in --data_dir it loads
    Data/2026/Synthetic_train_data_{with,without}_exp_{fm}.csv
and the matching
    model/2026/{fm}/{GB,RF,XGB}_exp_{Yes,No}.sav
(each a list of per-P fitted estimators, in sorted-P order), predicts the
'target' column and prints the in-sample RMSE and R2 per (FM, model, with_exp).

Combinations whose train CSV or .sav file is missing are skipped with a note.

Run from Helicity/Code:  python train_error.py
"""

import argparse
import glob
import os
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

pd.options.mode.chained_assignment = None

# with_exp_id -> the train-CSV name fragment written by prepare_train_test_data.py
FILE_SUFFIX = {'Yes': 'with_exp', 'No': 'without_exp'}
MODEL_TYPES = ['GB', 'RF', 'XGB']


def discover_fms(data_dir, with_exp_id):
    suffix = FILE_SUFFIX[with_exp_id]
    pattern = os.path.join(data_dir, f'Synthetic_train_data_{suffix}_*.csv')
    fms = []
    for path in sorted(glob.glob(pattern)):
        m = re.search(rf'Synthetic_train_data_{suffix}_(.+)\.csv$', os.path.basename(path))
        if m:
            fms.append(m.group(1))
    return fms


class Train_Error:
    def __init__(self, data_dir='../Data/2026', model_dir='../model/2026'):
        self.data_dir = data_dir
        self.model_dir = model_dir

    def _score_one(self, df_train, estimators):
        """estimators: list of per-P models in sorted-P order (as saved by
        model.py). Returns (rmse, r2, n) over the whole frame, predicting each
        P with its own estimator."""
        ordered_ps = sorted(df_train.P.unique())
        if len(estimators) != len(ordered_ps):
            raise ValueError(f'{len(estimators)} per-P models but {len(ordered_ps)} '
                             f'P values {ordered_ps}')
        per_p = dict(zip(ordered_ps, estimators))

        y_true, y_pred = [], []
        for p in ordered_ps:
            tmp = df_train.loc[df_train.P == p]
            X = tmp.iloc[:, :-1]                 # keep column names (models were fit on named frames)
            y_true.append(tmp['target'].to_numpy())
            y_pred.append(np.asarray(per_p[p].predict(X), dtype=float))

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        return rmse, float(r2_score(y_true, y_pred)), len(y_true)

    def compute_error(self):
        rows = []
        for with_exp_id, suffix in FILE_SUFFIX.items():
            for fm in discover_fms(self.data_dir, with_exp_id):
                train_csv = os.path.join(self.data_dir, f'Synthetic_train_data_{suffix}_{fm}.csv')
                df_train = pd.read_csv(train_csv)

                for model_type in MODEL_TYPES:
                    sav = os.path.join(self.model_dir, str(fm), f'{model_type}_exp_{with_exp_id}.sav')
                    if not os.path.exists(sav):
                        print(f'skip FM={fm} {model_type} exp_{with_exp_id}: {sav} not found')
                        continue

                    estimators = joblib.load(sav)
                    rmse, r2, n = self._score_one(df_train, estimators)
                    rows.append({'FM': fm, 'model': model_type, 'with_exp': with_exp_id,
                                 'n_rows': n, 'RMSE': rmse, 'R2': r2})
                    print(f'FM={fm:>3} {model_type:>3} exp_{with_exp_id:<3}  '
                          f'n={n:>8}  RMSE={rmse:.5f}  R2={r2:.5f}')

        if rows:
            df = pd.DataFrame(rows)
            out = os.path.join(self.data_dir, 'train_error_summary.csv')
            df.to_csv(out, index=False)
            print(f'\nsummary written -> {out}')
            print(df.to_string(index=False))
        else:
            print('no (train CSV, model) pairs found')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='../Data/2026',
                        help="Directory holding Synthetic_train_data_{with,without}_exp_{fm}.csv")
    parser.add_argument("--model_dir", default='../model/2026',
                        help="Directory holding {model_dir}/{fm}/{model_type}_exp_{id}.sav")
    args = parser.parse_args()

    te = Train_Error(data_dir=args.data_dir, model_dir=args.model_dir)
    te.compute_error()
