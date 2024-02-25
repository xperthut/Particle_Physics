import numpy as np
import pandas as pd
import math
import warnings
warnings.filterwarnings("ignore")

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.mode.chained_assignment = None

class Prepare_train_data:
    def __init__(self, name=''):
        self.name = name
        
    def train_data(self):
        df_real = pd.read_csv('../Data/All_p_w_m.csv')
        df_real.W = np.round(df_real.W, decimals=1)
        df_real.sort_values(['P','Exp','W'], ascending=True, inplace=True)
        df_real = df_real[['P','Exp','W','M']]
        
        df_data = pd.read_csv('../Data/generated_Synthetic_exp_data.csv')
        df_data = df_data[['P','Exp','W','M']]
        
        tmp_df = []
        for p in range(1,7):
            for exp in range(1,1902):
                tmp = df_data[np.logical_and(df_data.P==p, df_data.Exp==exp)]
                tmp.sort_values(['W'], ascending=True, inplace=True)
                tmp.reset_index(drop=True, inplace=True)

                for i in range(0, 197):
                    tmp_df.append([p,tmp.W[i],tmp.M[i],tmp.W[i+1],tmp.M[i+1],tmp.W[i+2],tmp.M[i+2],tmp.W[i+3],tmp.M[i+3],tmp.W[i+4],tmp.M[i+4]])

            print('{} done.'.format(p))

        df_train = pd.DataFrame(tmp_df,columns=['P','w1','m1','w2','m2','w3','m3','w4','m4','w5','target'])
        del [tmp_df]
        df_train.P = df_train.P.astype(int)
        df_train.to_csv('../Data/Synthetic_train_data_without_exp.csv', index=False)
        
        tmp_df = []
        for p in range(1,7):
            for exp in range(1,1902):
                tmp = df_data[np.logical_and(df_data.P==p, df_data.Exp==exp)]
                tmp.sort_values(['W'], ascending=True, inplace=True)
                tmp.reset_index(drop=True, inplace=True)

                for i in range(0, 197):
                    tmp_df.append([p,exp,tmp.W[i],tmp.M[i],tmp.W[i+1],tmp.M[i+1],tmp.W[i+2],tmp.M[i+2],tmp.W[i+3],tmp.M[i+3],tmp.W[i+4],tmp.M[i+4]])

            print('{} done.'.format(p))

        df_train = pd.DataFrame(tmp_df,columns=['P','Exp','w1','m1','w2','m2','w3','m3','w4','m4','w5','target'])
        del [tmp_df]
        df_train.P = df_train.P.astype(int)
        df_train.Exp = df_train.Exp.astype(int)
        df_train.to_csv('../Data/Synthetic_train_data_with_exp.csv', index=False)
        
if __name__=="__main__":
    tr = Prepare_train_data()
    tr.train_data()
