from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from sklearn import pipeline
import sklearn
import xgboost
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

# Bayesian-optimization iteration budget shared by every model's BayesSearchCV
# unless a caller overrides it per call.
DEFAULT_N_ITER = 20
# CV folds used during the search phase. Kept small because TimeSeriesSplit's
# later folds train on almost the full search sample each time, so cost scales
# ~linearly with this.
DEFAULT_CV_SPLITS = 5
# Hard cap on rows used *while searching* hyperparameters, since
# BayesSearchCV re-fits the estimator n_iter * cv_splits times and this
# pipeline's per-P train frames run into the hundreds of thousands of rows.
# The winning config is refit on the FULL per-P data afterwards, so this only
# trades a little search precision for a lot of speed, not final-model
# accuracy.
DEFAULT_SEARCH_SAMPLE_SIZE = 20000


def _to_native(value):
    """Cast numpy scalar types coming out of BayesSearchCV's best_params_ to
    plain python types so the params dict is JSON-serialisable."""
    if isinstance(value, np.generic):
        return value.item()
    return value


class Model:
    def __init__(self, name=''):
        self.model_name = name

    def _bayes_search(self, estimator, search_spaces, X, Y, p, n_iter, verbose,
                       cv_splits=DEFAULT_CV_SPLITS, search_sample_size=DEFAULT_SEARCH_SAMPLE_SIZE):
        # random_state is seeded per-P (not globally) so each P's search is
        # reproducible on its own but independent of the others, matching the
        # per-P seeding convention used elsewhere in this pipeline.
        X = X.reset_index(drop=True)
        n = len(X)

        if n > search_sample_size:
            # Sorted random subsample: cheap to search on, and sorting keeps
            # row order intact so TimeSeriesSplit's fold-by-order semantics
            # still mean roughly the same thing as on the full data.
            rng = np.random.RandomState(p)
            idx = np.sort(rng.choice(n, size=search_sample_size, replace=False))
            X_search, Y_search = X.iloc[idx], Y[idx]
        else:
            X_search, Y_search = X, Y

        start = time.time()
        splits = min(cv_splits, len(X_search) - 1)

        search = BayesSearchCV(
            estimator=estimator,
            search_spaces=search_spaces,
            n_iter=n_iter,
            cv=TimeSeriesSplit(n_splits=splits),
            scoring="neg_mean_squared_error",
            verbose=0,
            refit=False,  # we refit the winning config on the full data ourselves, below
            n_jobs=-1,
            random_state=p,
        )

        def _progress(res):
            if verbose > 0:
                done = len(res.func_vals)
                print('    [P={}] {}/{} evaluated, best MSE so far={:.4f}, elapsed={:.0f}s'.format(
                    p, done, n_iter, np.min(res.func_vals), time.time() - start), flush=True)

        search.fit(X_search, Y_search, callback=_progress if verbose > 0 else None)

        best_params = {k: _to_native(v) for k, v in search.best_params_.items()}

        # Refit the winning hyperparameters on the FULL per-P data (not just
        # the search subsample) so the saved model uses everything available.
        best_model = clone(estimator).set_params(**search.best_params_)
        best_model.fit(X, Y)

        if verbose > 0:
            print(best_params)
            print('[P={}] search done in {:.0f}s'.format(p, time.time() - start))

        return best_model, best_params

    def GradientBoostedModel(self, df_train, verbose=0, n_iter=DEFAULT_N_ITER, **search_kwargs):
        model = []
        best_params = {}

        if verbose > 0:
            print('////////////////////////////////////////////// Gradient boosted regressor ////////////////////////////////')

        for p in sorted(df_train.P.unique()):
            tmp = df_train.loc[df_train.P == p]
            X = tmp.iloc[:, :-1]
            Y = tmp.iloc[:, -1:].values.ravel()

            estimator = pipeline.Pipeline([
                ('gbc', GradientBoostingRegressor())
            ])

            search_spaces = {
                "gbc__n_estimators": Integer(10, 300),
                "gbc__learning_rate": Real(1e-3, 1.0, prior="log-uniform"),
                "gbc__max_depth": Integer(2, 10),
                "gbc__subsample": Real(0.5, 1.0),
            }

            bestModel, params = self._bayes_search(estimator, search_spaces, X, Y, p, n_iter, verbose, **search_kwargs)

            model.append(bestModel)
            best_params[int(p)] = params

            if verbose > 0:
                print(bestModel)
                print('{} done.'.format(p))

        return model, best_params

    def RandomForestModel(self, df_train, verbose=0, n_iter=DEFAULT_N_ITER, **search_kwargs):
        model = []
        best_params = {}

        if verbose > 0:
            print('////////////////////////////////////////////// Random Forest regressor ////////////////////////////////')

        for p in sorted(df_train.P.unique()):
            tmp = df_train.loc[df_train.P == p]
            X = tmp.iloc[:, :-1]
            Y = tmp.iloc[:, -1:].values.ravel()

            estimator = pipeline.Pipeline([
                ('rf', RandomForestRegressor(bootstrap=True))
            ])

            search_spaces = {
                "rf__n_estimators": Integer(2, 300),
                "rf__max_depth": Integer(2, 50),
                "rf__min_samples_split": Integer(2, 20),
                "rf__min_samples_leaf": Integer(1, 20),
            }

            bestModel, params = self._bayes_search(estimator, search_spaces, X, Y, p, n_iter, verbose, **search_kwargs)

            model.append(bestModel)
            best_params[int(p)] = params

            if verbose > 0:
                print(bestModel)
                print('{} done.'.format(p))

        return model, best_params

    def XGBoostModel(self, df_train, verbose=0, n_iter=DEFAULT_N_ITER, **search_kwargs):
        model = []
        best_params = {}

        if verbose > 0:
            print('////////////////////////////////////////////// XGBoost regressor ////////////////////////////////')

        for p in sorted(df_train.P.unique()):
            tmp = df_train.loc[df_train.P == p]
            X = tmp.iloc[:, :-1]
            Y = tmp.iloc[:, -1:].values.ravel()

            estimator = pipeline.Pipeline([
                ('xgb', XGBRegressor(colsample_bytree=0.8))
            ])

            search_spaces = {
                "xgb__n_estimators": Integer(10, 300),
                "xgb__max_depth": Integer(2, 50),
                "xgb__eta": Real(1e-3, 0.5, prior="log-uniform"),
                "xgb__subsample": Real(0.5, 0.8),
            }

            bestModel, params = self._bayes_search(estimator, search_spaces, X, Y, p, n_iter, verbose, **search_kwargs)

            model.append(bestModel)
            best_params[int(p)] = params

            if verbose > 0:
                print(bestModel)
                print('{} done.'.format(p))

        return model, best_params
