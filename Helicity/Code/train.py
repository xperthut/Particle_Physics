from model import Model
import joblib
import pandas as pd
import argparse
import glob
import json
import os
import re

# with_exp_id -> the train-CSV name fragment written by prepare_train_test_data.py
FILE_SUFFIX = {'Yes': 'with_exp', 'No': 'without_exp'}
MODEL_TYPES = ['GB', 'RF', 'XGB', 'NN', 'GP', 'BNN', 'NNE']


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
        elif self.name == 'NN':
            return self.model.NeuralNetworkModel(df_train, 1, n_iter, **search_kwargs)
        elif self.name == 'GP':
            return self.model.GaussianProcessModel(df_train, 1, n_iter, **search_kwargs)
        elif self.name == 'BNN':
            return self.model.BayesianNNModel(df_train, 1, n_iter, **search_kwargs)
        elif self.name == 'NNE':
            return self.model.NNEnsembleModel(df_train, 1, n_iter, **search_kwargs)

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
                         help="GB for Gradient boosted, RF for Random forest, XGB for Extreme gradient boosted, "
                              "NN for a neural network (MLP), GP for a Gaussian process, and BNN for a "
                              "Bayesian neural network sampled with HMC.")
    parser.add_argument("--with_exp_id", type=str, required=False, default="No",
                         help="'Yes' trains on Synthetic_train_data_with_exp_{fm}.csv, 'No' on the without_exp variant.")
    parser.add_argument("--data_dir", type=str, required=False, default="../Data/2026",
                         help="Directory holding Synthetic_train_data_{with,without}_exp_{fm}.csv")
    parser.add_argument("--model_dir", type=str, required=False, default="../model/2026",
                         help="Directory to save trained models/best hyperparameters under {model_dir}/{fm}/")
    parser.add_argument("--n_iter", type=int, required=False, default=20,
                         help="Number of BayesSearchCV iterations per P.")
    parser.add_argument("--cv_splits", type=int, required=False, default=5,
                         help="Forward-in-omega CV folds (OmegaBlockSplit on w5) used during the "
                              "hyperparameter search (fewer = faster).")
    parser.add_argument("--search_sample_size", type=int, required=False, default=20000,
                         help="Row cap used while searching hyperparameters per P; the winning config is "
                              "still refit on the full per-P data afterwards.")
    parser.add_argument("--gp_max_samples", type=int, required=False, default=2000,
                         help="GP only: rows per P the Gaussian process is fit on (exact GP cost grows "
                              "as n^3, so it is never fit on the full per-P data).")

    # BNN only (bnn.py / model.DEFAULT_BNN_PARAMS)
    parser.add_argument("--bnn_width", type=int, default=32,
                        help="BNN: first hidden layer width (layers halve, like NN)")
    parser.add_argument("--bnn_depth", type=int, default=2, help="BNN: number of hidden layers")
    parser.add_argument("--bnn_max_samples", type=int, default=3000,
                        help="BNN: rows per P the posterior is sampled on (full-batch HMC)")
    parser.add_argument("--hmc_chains", type=int, default=2, help="BNN: HMC chains per P")
    parser.add_argument("--hmc_warmup", type=int, default=600,
                        help="BNN: warmup iterations per chain (step size + mass matrix tuning)")
    parser.add_argument("--hmc_samples", type=int, default=600,
                        help="BNN: sampling iterations per chain")
    parser.add_argument("--hmc_leapfrog", type=int, default=30,
                        help="BNN: leapfrog steps per HMC iteration")
    parser.add_argument("--hmc_keep", type=int, default=200,
                        help="BNN: posterior draws kept in the saved model (over all chains)")

    # NNE only (model.NNEnsembleRegressor): bootstrap ensemble of small MLPs
    parser.add_argument("--nne_members", type=int, default=16, help="NNE: ensemble members per P")
    parser.add_argument("--nne_width", type=int, default=8, help="NNE: first hidden layer width")
    parser.add_argument("--nne_depth", type=int, default=1, help="NNE: number of hidden layers")
    parser.add_argument("--nne_alpha", type=float, default=1e-3, help="NNE: L2 penalty")
    parser.add_argument("--nne_rows", type=int, default=50000,
                        help="NNE: max rows per member after the replica bootstrap")

    args = parser.parse_args()

    if args.model_type not in MODEL_TYPES:
        print("Invalid model type.")
    elif args.with_exp_id not in FILE_SUFFIX:
        print("Invalid with_exp_id, must be 'Yes' or 'No'.")
    else:
        fms = [args.fm] if args.fm != "all" else discover_fms(args.data_dir, args.with_exp_id)

        if not fms:
            print(f"No Synthetic_train_data_{FILE_SUFFIX[args.with_exp_id]}_*.csv files found in {args.data_dir}")

        if args.model_type == 'GP':
            search_kwargs = {'gp_max_samples': args.gp_max_samples}
        elif args.model_type == 'BNN':
            search_kwargs = {'width': args.bnn_width, 'depth': args.bnn_depth,
                             'max_samples': args.bnn_max_samples, 'n_chains': args.hmc_chains,
                             'n_warmup': args.hmc_warmup, 'n_samples': args.hmc_samples,
                             'n_leapfrog': args.hmc_leapfrog, 'n_keep': args.hmc_keep}
        elif args.model_type == 'NNE':
            search_kwargs = {'n_members': args.nne_members, 'width': args.nne_width,
                             'depth': args.nne_depth, 'alpha': args.nne_alpha,
                             'n_rows': args.nne_rows}
        else:
            search_kwargs = {'cv_splits': args.cv_splits, 'search_sample_size': args.search_sample_size}

        for fm in fms:
            train_data = os.path.join(args.data_dir, f'Synthetic_train_data_{FILE_SUFFIX[args.with_exp_id]}_{fm}.csv')

            if not os.path.exists(train_data):
                print(f"{train_data} doesn't exist")
                continue

            df_train = pd.read_csv(train_data)
            tr = Train(args.model_type)
            tr.train_model(df_train, fm, args.with_exp_id, args.model_dir, args.n_iter, search_kwargs)
