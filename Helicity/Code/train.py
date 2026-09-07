from model import Model
import joblib
import pandas as pd
import numpy as np
import argparse
import glob
import json
import os
import re

# with_exp_id -> the train-CSV name fragment written by prepare_train_test_data.py
FILE_SUFFIX = {'Yes': 'with_exp', 'No': 'without_exp'}


class Train:
    def __init__(self, name):
        self.name = name
        self.model = Model()

    def _fit(self, df_train, n_iter, search_kwargs):
        if self.name == 'GB':
            return self.model.GradientBoostedModel(df_train, 1, n_iter, **search_kwargs)
        elif self.name == 'RF':
            return self.model.RandomForestModel(df_train, 1, n_iter, **search_kwargs)
        elif self.name == 'XGB':
            return self.model.XGBoostModel(df_train, 1, n_iter, **search_kwargs)

    def train_model(self, df_train, fm, with_exp_id, model_dir, n_iter, search_kwargs):
        save_name = f"{self.name}_exp_{with_exp_id}"
        print(f"FM={fm} {save_name}")

        model, best_params = self._fit(df_train, n_iter, search_kwargs)

        out_dir = os.path.join(model_dir, str(fm))
        os.makedirs(out_dir, exist_ok=True)

        joblib.dump(model, os.path.join(out_dir, f'{save_name}.sav'))
        with open(os.path.join(out_dir, f'{save_name}_best_params.json'), 'w') as f:
            json.dump(best_params, f, indent=2)

        print(f'FM={fm} {save_name} done and saved to {out_dir}')


def discover_fms(data_dir, with_exp_id):
    suffix = FILE_SUFFIX[with_exp_id]
    pattern = os.path.join(data_dir, f'Synthetic_train_data_{suffix}_*.csv')
    fms = []
    for path in sorted(glob.glob(pattern)):
        m = re.search(rf'Synthetic_train_data_{suffix}_(.+)\.csv$', os.path.basename(path))
        if m:
            fms.append(m.group(1))
    return fms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fm", type=str, required=False, default="all",
                         help="Ensemble id (e.g. '9', '12', '15'), or 'all' to train every FM found in --data_dir.")
    parser.add_argument("--model_type", type=str, required=True,
                         help="GB for Gradient boosted, RF for Random forest, and XGB for Extreme gradient boosted.")
    parser.add_argument("--with_exp_id", type=str, required=False, default="No",
                         help="'Yes' trains on Synthetic_train_data_with_exp_{fm}.csv, 'No' on the without_exp variant.")
    parser.add_argument("--data_dir", type=str, required=False, default="../Data/2026",
                         help="Directory holding Synthetic_train_data_{with,without}_exp_{fm}.csv")
    parser.add_argument("--model_dir", type=str, required=False, default="../model/2026",
                         help="Directory to save trained models/best hyperparameters under {model_dir}/{fm}/")
    parser.add_argument("--n_iter", type=int, required=False, default=20,
                         help="Number of BayesSearchCV iterations per P.")
    parser.add_argument("--cv_splits", type=int, required=False, default=5,
                         help="TimeSeriesSplit folds used during the hyperparameter search (fewer = faster).")
    parser.add_argument("--search_sample_size", type=int, required=False, default=20000,
                         help="Row cap used while searching hyperparameters per P; the winning config is "
                              "still refit on the full per-P data afterwards.")

    args = parser.parse_args()

    if args.model_type not in ['GB', 'XGB', 'RF']:
        print("Invalid model type.")
    elif args.with_exp_id not in FILE_SUFFIX:
        print("Invalid with_exp_id, must be 'Yes' or 'No'.")
    else:
        fms = [args.fm] if args.fm != "all" else discover_fms(args.data_dir, args.with_exp_id)

        if not fms:
            print(f"No Synthetic_train_data_{FILE_SUFFIX[args.with_exp_id]}_*.csv files found in {args.data_dir}")

        search_kwargs = {'cv_splits': args.cv_splits, 'search_sample_size': args.search_sample_size}

        for fm in fms:
            train_data = os.path.join(args.data_dir, f'Synthetic_train_data_{FILE_SUFFIX[args.with_exp_id]}_{fm}.csv')

            if not os.path.exists(train_data):
                print(f"{train_data} doesn't exist")
                continue

            df_train = pd.read_csv(train_data)
            tr = Train(args.model_type)
            tr.train_model(df_train, fm, args.with_exp_id, args.model_dir, args.n_iter, search_kwargs)
