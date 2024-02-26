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

class Reconstruct_real_data:
    def __init__(self):
        pass
    
    def reconstruct_data(self):
        df_real = pd.read_csv('../Data/All_p_w_m.csv')
        df_real.W = np.round(df_real.W, decimals=1)
        df_real.sort_values(['P','Exp','W'], ascending=True, inplace=True)
        df_real = df_real[['P','Exp','W','M']]
        
        df_data = pd.read_csv('../Data/generated_Synthetic_exp_data.csv')
        
        real_filled = []

        df_data.sort_values(['P','Exp','W'], ascending=True, inplace=True)
        df_real.sort_values(['P','Exp','W'], ascending=True, inplace=True)

        for p in range(1,7):
            for exp in range(1, 1901):
                gen = df_data.loc[np.logical_and(df_data.P==p, df_data.Exp==exp), ['W','M']]
                real = df_real.loc[np.logical_and(df_real.P==p, df_real.Exp==exp), ['W','M']]

                i=0
                for w in gen.W.unique():
                    i+=1
                    if i==5:
                        break

                    m = gen.M[gen.W==w].values[0]
                    if w in real.W:
                        m = real.M[real.W==w].values[0]

                    real_filled.append([p, exp, w, m])

        df_real_filled = pd.DataFrame(real_filled, columns=['P','Exp','W','M'])
        
        test = []
        allWs = sorted(np.round(df_data.W.unique(), decimals=1))
        loaded_model = [
            joblib.load('../model/GB_exp_Yes.sav'), 
            joblib.load('../model/RF_exp_Yes.sav'), 
            joblib.load('../model/XGB_exp_Yes.sav')
        ]

        for p in range(1,7):
            for exp in range(1, 1901):
                tmpGB = []
                tmpRF = []
                tmpXGB = []

                W = sorted(df_real_filled.W[np.logical_and(df_real_filled.P==p, df_real_filled.Exp==exp)].unique())

                for w in W[:4]:
                    m = df_real_filled.M[np.logical_and(df_real_filled.P==p, np.logical_and(df_real_filled.Exp==exp, df_real_filled.W==w))].values[0]
                    test.append([p,exp,w,m,m,m])

                    tmpGB.append(w)
                    tmpGB.append(m)
                    tmpRF.append(w)
                    tmpRF.append(m)
                    tmpXGB.append(w)
                    tmpXGB.append(m)

                #print('real m=', df_real.M[np.logical_and(df_real.P==p, np.logical_and(df_real.Exp==exp, df_real.W==W[4]))].values[0])

                term = 1
                W_new = [x for x in allWs if x>W[3]]
                #print(len(W_new))
                for w in W_new:
                    #print(w)
                    b = [p, exp, w]

                    tmpGB.append(w)
                    tmpRF.append(w)
                    tmpXGB.append(w)

                    a = [p,exp]
                    a.extend(tmpGB)
                    a = np.array(a).reshape(1,-1)
                    #print(a)
                    m = loaded_model[0][p-1].predict(a)[0]
                    #print('m=',m)
                    b.append(m)
                    tmpGB.append(m)

                    a = [p,exp]
                    a.extend(tmpRF)
                    a = np.array(a).reshape(1,-1)
                    #print(a)
                    m = loaded_model[1][p-1].predict(a)[0]
                    #print('m=',m)
                    b.append(m)
                    tmpRF.append(m)

                    a = [p,exp]
                    a.extend(tmpXGB)
                    a = np.array(a).reshape(1,-1)
                    #print(a)
                    m = loaded_model[2][p-1].predict(a)[0]
                    #print('m=',m)
                    b.append(m)
                    tmpXGB.append(m)

                    #print(tmpLR)
                    #print(tmpGB)
                    #print(tmpRF)
                    #print(tmpXGB)

                    test.append(b)

                    tmpGB.pop(0)
                    tmpGB.pop(0)

                    tmpRF.pop(0)
                    tmpRF.pop(0)

                    tmpXGB.pop(0)
                    tmpXGB.pop(0)

                    #term += 1

                    #if term==3:
                     #   break

                    #print('\n\n')

            print(p, 'Done')

        #test
        df_test = pd.DataFrame(test, columns=['P','Exp','W','M_GB','M_RF','M_XGB'])
        del [test]
        df_test.to_csv('../Data/Real_data_projected_v2.csv', index = False)
        print('Saving done...')
        
        
if __name__=="__main__":
    rd = Reconstruct_real_data()
    rd.reconstruct_data()