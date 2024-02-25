import numpy as np
import pandas as pd
import altair as alt
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from numpy.random import seed
from numpy.random import normal

warnings.filterwarnings("ignore")

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.mode.chained_assignment = None

class Generator:
    def __init__(self, name=''):
        self.name = name
        
    def prepare_real_data(self):
        n = 1901
        df = None

        for i in range(1, 7):
            tmp = pd.DataFrame({'P':[i]*n, 'W':[0]*n, 'Exp':[i for i in range(1, n+1)], 'M':[0]*n})

            if df is None:
                df = pd.DataFrame(tmp)
            else:
                df = pd.concat([df, tmp], ignore_index=True)

            del [tmp]

        for p in range(1, 7):
            tmp = pd.read_csv('../Data/p{}_pnu.txt'.format(p), sep=' ', header=None)
            w = list(tmp.iloc[0, :].values)
            tmp = tmp.iloc[1:, :]
            tmp=tmp.T
            tmp['P'] = [p]*8
            tmp['W'] = w
            col = ['P',  'W']
            col.extend([i for i in range(1, n+1)])
            tmp = tmp[col]
            tmp = tmp.melt(id_vars=['P','W'], value_vars=[i for i in range(1, n+1)], var_name='Exp', value_name='M')
            tmp = tmp[['P','W','Exp','M']]

            df = pd.concat([df, tmp], ignore_index=True)
            del [tmp]

        df.sort_values(['P','W','Exp', 'M'], ascending=True, inplace=True)

        df.to_csv('../Data/All_p_w_m.csv', index=False)
        
    def generate_M_using_gradient_descent(self):
        df_I = pd.read_csv('../Data/ITD-pol-fit1-line.txt', header=None, sep=' ')
        df_I.columns = ['W', 'I']
        
        A,B = -1/3, 4/3
        coeff = [A,B]
        lr = 0.01
        loss_val={'epoch':list(),'w':list(), 'loss':list()}
        Epoc=100
        t = []

        def predict(x):
            return ((A*x[0])+(B*x[1]))

        def lossFnc(f,y):
            return ((f-y)**2)

        for i in df_I.index:
            W = df_I.W[i]
            Yreal = df_I.I[i]
            #m1=290.0
            m1,m2 = 310.0*Yreal, 0.0 

            for ep in range(1, Epoc):
                Ypred=predict([m1,m2])
                loss = lossFnc(Yreal,Ypred)

                loss_val['epoch'].append(ep)
                loss_val['w'].append(W)
                loss_val['loss'].append(loss)

                dm1=-2*A*(Yreal-Ypred)
                dm2=-2*B*(Yreal-Ypred)

                m1=m1-(lr*dm1)
                m2=m2-(lr*dm2)

            t.append({
                'W':W,
                'M1':m1,
                'M2':m2,
                'M3':((8*Yreal)+m1)/9,
                'M4':((15*Yreal)+m1)/16,
                'M5':((24*Yreal)+m1)/25,
                'M6':((35*Yreal)+m1)/36
            })

        pd.DataFrame(loss_val).to_csv('../Data/loss_val.csv', index=False)

        df_M=pd.DataFrame(t)
        df_data = df_M.melt(id_vars='W', value_vars=['M1','M2','M3','M4','M5','M6'], var_name='P', value_name='Est_M')
        df_data.P = [int(x[1]) for x in df_data.P]
        print(df_data.shape)
        df_data.to_csv('../Data/new_M.csv', index=False)
        
    def __regression_analysis_on_std__(self):
        df_real = pd.read_csv('../Data/All_p_w_m.csv')
        model_LR=[]

        for p in range(1,7):
            w1 = df_real.W[df_real.P==p].unique()
            list_w, list_std = [],[]
            for w in w1:#[:-4]:
                t = df_real[np.logical_and(df_real.W==w, df_real.P==p)]
                list_w.append(w)
                list_std.append(np.std(t.M))

            tmp = pd.DataFrame({'W':list_w, 'S':list_std})
            X = np.array(tmp.W).reshape(-1, 1)
            y = tmp.S.values.ravel()

            bestModel = LinearRegression(fit_intercept=False,).fit(X, y)
            pred_std = bestModel.predict(X)

            model_LR.append(bestModel)
            
        return model_LR
    
    def generate_data(self):
        model_LR = self.__regression_analysis_on_std__()
        
        df_data = pd.read_csv('../Data/new_M.csv')
        df_data.W = np.round(df_data.W,1)
        
        df_data['Est_S'] = 0.0
        for p in df_data.P.unique():
            df_data.loc[df_data.P==p, 'Est_S'] = model_LR[p-1].predict(df_data.W[df_data.P==p].values.reshape(-1, 1))
            #print(len(df_data.W), len(stdVal), stdVal[:10])

        df_data.Est_S[df_data.W==0.0] = 0.0
        #df_M.to_csv('../Data/new_M_S.csv', index=False)
        
        #df_data = pd.read_csv('../Data/new_M_S.csv')
        cols = []
        for i in range(1, 1902):
            c = f'E{i}'
            cols.append(c)
            df_data.loc[:,c] = df_data.Est_M

        df_exp_data = df_data.melt(id_vars=['W','P','Est_M','Est_S'], value_vars=cols, value_name='M', var_name='Exp')
        del[df_data]
        df_exp_data.Exp = [int(x[1:]) for x in df_exp_data.Exp]
        
        for p in df_exp_data.P.unique():
            seed(p)

            for w in df_exp_data.W[df_exp_data.P==p].unique():
                t = df_exp_data.loc[np.logical_and(df_exp_data.W==w, df_exp_data.P==p), ['Est_M', 'Est_S']].drop_duplicates()
                mu = t.Est_M.values[0]
                sigma = t.Est_S.values[0]

                #generate sample of 1901 values that follow a normal distribution 
                #expData = normal(loc=mu, scale=sigma, size=1901)
                df_exp_data.loc[np.logical_and(df_exp_data.W==w, df_exp_data.P==p), 'M'] = normal(loc=mu, scale=sigma, size=1901)
                #if len(set(expData)) != 1901:
                #    print(p, w, mu, sigma, len(expData), len(set(expData)))

        df_exp_data.to_csv('../Data/generated_Synthetic_exp_data.csv', index=False)
        
if __name__=="__main__":
    gen = Generator()
    gen.prepare_real_data()
    gen.generate_M_using_gradient_descent()
    gen.generate_data()
    print('Data generation done...')