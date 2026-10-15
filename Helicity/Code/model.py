from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn import pipeline
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
# CV folds used during the search phase (OmegaBlockSplit, forward-chaining in
# omega). Kept small because later folds train on most of the search sample
# each time, so cost scales ~linearly with this.
DEFAULT_CV_SPLITS = 5
# Hard cap on rows used *while searching* hyperparameters, since
# BayesSearchCV re-fits the estimator n_iter * cv_splits times and this
# pipeline's per-P train frames run into the hundreds of thousands of rows.
# The winning config is refit on the FULL per-P data afterwards, so this only
# trades a little search precision for a lot of speed, not final-model
# accuracy.
DEFAULT_SEARCH_SAMPLE_SIZE = 20000
# Rows a Gaussian process is fit on per P. Exact GP fitting is O(n^3) in time
# and O(n^2) in memory, so unlike the other models it cannot be refit on the
# full (~200k-row) per-P frame; it is fit on a seeded random subsample instead.
DEFAULT_GP_MAX_SAMPLES = 2000


class ChunkedGaussianProcessRegressor(GaussianProcessRegressor):
    """GaussianProcessRegressor whose predict() runs in row chunks.

    Plain predict() builds the full (n_query x n_train) kernel matrix at once,
    which for a ~200k-row per-P train frame (train_error.py) is several GB.
    Chunking keeps memory bounded without changing the result.
    """
    PREDICT_CHUNK = 20000

    def predict(self, X, return_std=False, return_cov=False):
        if return_cov or len(X) <= self.PREDICT_CHUNK:
            return super().predict(X, return_std=return_std, return_cov=return_cov)
        X = np.asarray(X, dtype=float)
        parts = [super(ChunkedGaussianProcessRegressor, self).predict(
                     X[i:i + self.PREDICT_CHUNK], return_std=return_std)
                 for i in range(0, len(X), self.PREDICT_CHUNK)]
        if return_std:
            return (np.concatenate([m for m, _ in parts]),
                    np.concatenate([s for _, s in parts]))
        return np.concatenate(parts)


class ScaledMLPRegressor(BaseEstimator, RegressorMixin):
    """MLP with standardised inputs *and* target, sized by (width, depth).

    M spans orders of magnitude across P, so the target is standardised too
    (MLPRegressor does not do that itself). Layers halve in width:
    width=64, depth=3 -> (64, 32, 16). Exposing width/depth as plain integers
    lets BayesSearchCV search the architecture.
    """

    def __init__(self, width=32, depth=2, alpha=1e-3, learning_rate_init=1e-3,
                 batch_size=256, max_iter=300, random_state=0):
        self.width = width
        self.depth = depth
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.random_state = random_state

    def _hidden(self):
        return tuple(max(int(self.width) // 2 ** k, 4) for k in range(int(self.depth)))

    def fit(self, X, y):
        mlp = MLPRegressor(hidden_layer_sizes=self._hidden(), activation='relu',
                           alpha=self.alpha, learning_rate_init=self.learning_rate_init,
                           batch_size=self.batch_size, max_iter=self.max_iter,
                           early_stopping=True, n_iter_no_change=15,
                           random_state=self.random_state)
        self.model_ = pipeline.Pipeline([
            ('scale', StandardScaler()),
            ('mlp', TransformedTargetRegressor(regressor=mlp, transformer=StandardScaler())),
        ])
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class OmegaBlockSplit:
    """Forward-chaining CV over omega instead of row order.

    The train frames are ordered P -> Exp -> W, so TimeSeriesSplit folds end
    up splitting by Exp replica: validation rows are other noisy draws of the
    same curve on the same omega grid, which scores interpolation. The models
    are used to extrapolate forward in omega (reconstruct_real_data.py), so
    each fold here trains on windows whose target omega (``w_col``) is at or
    below a cut and validates on the next omega block above it. Cuts sit at
    quantiles of the distinct omega values, giving n_splits + 1 blocks.
    """

    def __init__(self, n_splits=DEFAULT_CV_SPLITS, w_col='w5'):
        self.n_splits = n_splits
        self.w_col = w_col

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        w = np.asarray(X[self.w_col], dtype=float)
        edges = np.quantile(np.unique(w), np.linspace(0, 1, self.n_splits + 2))
        for k in range(1, self.n_splits + 1):
            train = np.where(w <= edges[k])[0]
            test = np.where((w > edges[k]) & (w <= edges[k + 1]))[0]
            if len(train) and len(test):
                yield train, test


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
            # Random subsample: cheap to search on. OmegaBlockSplit folds by
            # w5, not row order, so the subsample keeps the same fold layout.
            rng = np.random.RandomState(p)
            idx = np.sort(rng.choice(n, size=search_sample_size, replace=False))
            X_search, Y_search = X.iloc[idx], Y[idx]
        else:
            X_search, Y_search = X, Y

        start = time.time()
        splits = min(cv_splits, X_search['w5'].nunique() - 1)

        search = BayesSearchCV(
            estimator=estimator,
            search_spaces=search_spaces,
            n_iter=n_iter,
            cv=list(OmegaBlockSplit(n_splits=splits).split(X_search)),
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

    def NeuralNetworkModel(self, df_train, verbose=0, n_iter=DEFAULT_N_ITER, **search_kwargs):
        model = []
        best_params = {}

        if verbose > 0:
            print('////////////////////////////////////////////// Neural network (MLP) regressor ////////////////////////////////')

        for p in sorted(df_train.P.unique()):
            tmp = df_train.loc[df_train.P == p]
            X = tmp.iloc[:, :-1]
            Y = tmp.iloc[:, -1:].values.ravel()

            estimator = pipeline.Pipeline([
                ('nn', ScaledMLPRegressor(random_state=int(p)))
            ])

            search_spaces = {
                "nn__width": Integer(16, 128),
                "nn__depth": Integer(1, 3),
                "nn__alpha": Real(1e-6, 1e-1, prior="log-uniform"),
                "nn__learning_rate_init": Real(1e-4, 1e-2, prior="log-uniform"),
            }

            bestModel, params = self._bayes_search(estimator, search_spaces, X, Y, p, n_iter, verbose, **search_kwargs)

            model.append(bestModel)
            best_params[int(p)] = params

            if verbose > 0:
                print(bestModel)
                print('{} done.'.format(p))

        return model, best_params

    def GaussianProcessModel(self, df_train, verbose=0, n_iter=DEFAULT_N_ITER,
                             gp_max_samples=DEFAULT_GP_MAX_SAMPLES, **search_kwargs):
        """Per-P GP with an ARD RBF + white-noise kernel.

        No BayesSearchCV here: the kernel hyperparameters (amplitude, one
        length scale per feature, noise level) are fit by maximising the
        marginal likelihood, which is how GPs are normally tuned. n_iter and
        the search kwargs are accepted only so train.py can call every model
        the same way. Each P is fit on at most ``gp_max_samples`` rows (seeded
        per P), not on the full frame -- see DEFAULT_GP_MAX_SAMPLES.
        """
        model = []
        best_params = {}

        if verbose > 0:
            print('////////////////////////////////////////////// Gaussian process regressor ////////////////////////////////')

        for p in sorted(df_train.P.unique()):
            tmp = df_train.loc[df_train.P == p].reset_index(drop=True)
            if len(tmp) > gp_max_samples:
                rng = np.random.RandomState(int(p))
                tmp = tmp.iloc[np.sort(rng.choice(len(tmp), size=gp_max_samples, replace=False))]
            X = tmp.iloc[:, :-1]
            Y = tmp.iloc[:, -1:].values.ravel()

            kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                      * RBF(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e3))
                      + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-8, 1.0)))

            bestModel = pipeline.Pipeline([
                ('scale', StandardScaler()),
                ('gp', ChunkedGaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                                       n_restarts_optimizer=2,
                                                       random_state=int(p))),
            ])

            start = time.time()
            bestModel.fit(X, Y)
            fitted = bestModel.named_steps['gp']

            best_params[int(p)] = {
                'kernel': str(fitted.kernel_),
                'log_marginal_likelihood': float(fitted.log_marginal_likelihood_value_),
                'n_fit_rows': int(len(X)),
            }
            model.append(bestModel)

            if verbose > 0:
                print(best_params[int(p)])
                print('[P={}] GP fit on {} rows in {:.0f}s'.format(p, len(X), time.time() - start))

        return model, best_params
