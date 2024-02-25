from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn import pipeline
import sklearn
import xgboost
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

class Model:
    def __init__(self, name=''):
        self.model_name = name
        
    def LinearRegressionModel(self, df_train, verbose=0):
        model = []

        if verbose>0:
            print('////////////////////////////////////////////// Linear regressor ////////////////////////////////')

        for p in range(1,7):
            tmp = df_train[df_train.P==p]
            X = tmp.iloc[:, :-1]
            Y = tmp.iloc[:,-1:].values.ravel()

            estimator = pipeline.Pipeline([
                ('poly', PolynomialFeatures(include_bias=False)),
                ('linear', LinearRegression())
            ])

            param_grid = {
                "poly__degree": [1,2],
                "linear__fit_intercept": [True, False],
                "linear__copy_X": [True, False],
                "linear__positive": [True, False]
            }

            grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=TimeSeriesSplit(n_splits=10),
                scoring="neg_mean_squared_error",
                verbose=0,
                refit=True, 
                n_jobs=-1
            )

            grid_search.fit(X, Y)

            if verbose>0:
                print(grid_search.best_params_)
                print('Train RMS={:.3f}'.format(grid_search.best_score_))

            bestModel = grid_search.best_estimator_

            if verbose>0:
                print(bestModel)

            model.append(bestModel)

            if verbose>0:
                print('{} done.'.format(p))

        return model

    def GradientBoostedModel(self, df_train, verbose=0):
        model = []

        if verbose>0:
            print('////////////////////////////////////////////// Gradient boosted regressor ////////////////////////////////')

        for p in range(1,7):
            tmp = df_train.loc[df_train.P==p]
            X = tmp.iloc[:, :-1]
            Y = tmp.iloc[:,-1:].values.ravel()

            estimator = pipeline.Pipeline([
                ('gbc', GradientBoostingRegressor())
            ])

            param_grid = {
                "gbc__n_estimators": [10, 20, 50, 100, 150, 200, 250, 300],
                "gbc__learning_rate": [0.001, 0.01, 0.1, 1.0]
            }

            grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=TimeSeriesSplit(n_splits=10),
                scoring="neg_mean_squared_error",
                verbose=0,
                refit=True, 
                n_jobs=-1
            )

            grid_search.fit(X, Y)

            if verbose>0:
                print(grid_search.best_params_)
                print('Train MSE={:.3f}'.format(grid_search.best_score_))

            bestModel = grid_search.best_estimator_

            if verbose>0:
                print(bestModel)

            model.append(bestModel)

            if verbose>0:
                print('{} done.'.format(p))

        return model

    def RandomForestModel(self, df_train, verbose=0):
        model = []

        if verbose>0:
            print('////////////////////////////////////////////// Random Forest regressor ////////////////////////////////')

        for p in range(1,7):
            tmp = df_train.loc[df_train.P==p]
            X = tmp.iloc[:, :-1]
            Y = tmp.iloc[:,-1:].values.ravel()

            estimator = pipeline.Pipeline([
                ('rf', RandomForestRegressor(bootstrap=True))
            ])

            param_grid = {
                "rf__n_estimators": [2, 4, 5, 10, 20, 50, 100, 150, 200],
                "rf__max_depth":[2,4,5,10,20,50]
            }

            grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=TimeSeriesSplit(n_splits=10),
                scoring="neg_mean_squared_error",
                verbose=0,
                refit=True, 
                n_jobs=-1
            )

            grid_search.fit(X, Y)

            if verbose>0:
                print(grid_search.best_params_)
                print('Train MSE={:.3f}'.format(grid_search.best_score_))

            bestModel = grid_search.best_estimator_

            if verbose>0:
                print(bestModel)

            model.append(bestModel)

            if verbose>0:
                print('{} done.'.format(p))

        return model

    def XGBoostModel(self, df_train, verbose=0):
        model = []

        if verbose>0:
            print('////////////////////////////////////////////// XGBoost regressor ////////////////////////////////')

        for p in range(1,7):
            tmp = df_train.loc[df_train.P==p]
            X = tmp.iloc[:, :-1]
            Y = tmp.iloc[:,-1:].values.ravel()

            estimator = pipeline.Pipeline([
                ('xgb', XGBRegressor(colsample_bytree=0.8))
            ])


            param_grid = {
                "xgb__n_estimators": [10, 20, 50, 100, 150, 200, 250, 300],
                "xgb__max_depth":[2,4,5,10,20,50],
                "xgb__eta":[0.001,0.01,0.1],
                "xgb__subsample":[0.5,0.7,0.8]
            }


            grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=TimeSeriesSplit(n_splits=10),
                scoring="neg_mean_squared_error",
                verbose=0,
                refit=True, 
                n_jobs=-1
            )

            grid_search.fit(X, Y)

            if verbose>0:
                print(grid_search.best_params_)
                print('Train MSE={:.3f}'.format(grid_search.best_score_))

            bestModel = grid_search.best_estimator_

            if verbose>0:
                print(bestModel)

            model.append(bestModel)

            if verbose>0:
                print('{} done.'.format(p))

        return model

    def SupportVectorModel(self, df_train, verbose=0):
        model = []

        if verbose>0:
            print('////////////////////////////////////////////// Support vector regressor ////////////////////////////////')

        for p in range(1,7):
            tmp = df_train.loc[df_train.P==p]
            X = tmp.iloc[:, :-1]
            Y = tmp.iloc[:,-1:].values.ravel()

            estimator = pipeline.Pipeline([
                ('svr', SVR(cache_size=1000))
            ])


            param_grid = {
                "svr__C": [0.01, 0.1, 1.0, 10.0],
                "svr__kernel": ['poly','rbf'],
                "svr__degree":[1,2]
            }


            grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=TimeSeriesSplit(n_splits=10),
                scoring="neg_mean_squared_error",
                verbose=0,
                refit=True, 
                n_jobs=-1
            )

            grid_search.fit(X, Y)

            if verbose>0:
                print(grid_search.best_params_)
                print('Train MSE={:.3f}'.format(grid_search.best_score_))

            bestModel = grid_search.best_estimator_

            if verbose>0:
                print(bestModel)

            model.append(bestModel)

            if verbose>0:
                print('{} done.'.format(p))

        return model