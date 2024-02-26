from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn import pipeline
import sklearn
import xgboost
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
import math
from sklearn.metrics import r2_score, mean_squared_error
from numpy.random import seed
from numpy.random import normal
import joblib

import warnings
warnings.filterwarnings("ignore")

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.mode.chained_assignment = None

class Train_Error:
    def __init__(self):
        self.data_dict = {
            'Yes':'../Data/Synthetic_train_data_with_exp.csv',
            'No':'../Data/Synthetic_train_data_without_exp.csv'
        }

        self.model_dict = {
            'GB':{
                'Yes':'../model/GB_exp_Yes.sav',
                'No':'../model/GB_exp_No.sav'
            },
            'RF':{
                'Yes':'../model/RF_exp_Yes.sav',
                'No':'../model/RF_exp_No.sav'
            },
            'XGB':{
                'Yes':'../model/XGB_exp_Yes.sav',
                'No':'../model/XGB_exp_No.sav'
            }
        }
        
    def compute_error(self):
        for key in self.data_dict:
            print(key)
            data = pd.read_csv(self.data_dict[key])

            for model_type in ['GB', 'RF', 'XGB']:
                model = joblib.load(self.model_dict[model_type][key])

                rv=[]
                pv=[]

                if key=='Yes':
                    for p in range(1,7):
                        #for exp in range(1, 1901):
                        X = data[data.P==p].iloc[:, :-1].values
                        rv.extend(data[data.P==p]['target'].values)
                        pv.extend(model[p-1].predict(X))

                elif key=='No':
                    for p in range(1,7):
                        X = data[data.P==p].iloc[:, [0,2,3,4,5,6,7,8,9,10]].values
                        rv.extend(data[data.P==p]['target'].values)
                        pv.extend(model[p-1].predict(X))

                print(data.shape[0], len(rv), len(pv))

                if len(rv)==data.shape[0] and len(pv)==data.shape[0]:
                    rmse = np.sqrt(mean_squared_error(rv,pv))
                    print(f"Model={model_type}, useExp={key}, RMSE={rmse:.4f}")
                
if __name__=="__main__":
    te = Train_Error()
    te.compute_error()
