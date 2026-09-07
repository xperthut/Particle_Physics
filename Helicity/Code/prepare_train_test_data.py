import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.mode.chained_assignment = None

class Prepare_train_data:
    def __init__(self, name='', Data_dir=''):
        self.name = name
        self.Data_dir = Data_dir
        self.Image_dir = os.path.join(self.Data_dir, 'Images')
        os.makedirs(self.Data_dir, exist_ok=True)
        os.makedirs(self.Image_dir, exist_ok=True)
        
    # 4 lag (W,M) points predicting the M of a 5th point ('target'); its W
    # ('w5') is kept as a feature since W sits on a fixed, known grid.
    WINDOW = 5

    def _iter_p_exp_groups(self, df_fm):
        """Yield (p, exp, tmp) for every (P,Exp) combination actually present
        in this FM's data, sorted by W within each group. P and Exp values
        (and how many of each exist) are read off the data itself rather
        than assumed, since different P values can have different Exp
        coverage.
        """
        for p in sorted(df_fm.P.unique()):
            df_p = df_fm[df_fm.P == p].sort_values('W')
            for exp, tmp in df_p.groupby('Exp', sort=True):
                yield p, exp, tmp.reset_index(drop=True)

    def train_data(self):
        df_data = pd.read_csv(os.path.join(self.Data_dir, 'generated_Synthetic_exp_data.csv'))
        df_data = df_data[['FM','P','Exp','W','M']]

        # Each FM is its own physical series, so it gets its own raw-data
        # split and its own pair of train CSVs (P alone, no FM column, is
        # then a safe per-file grouping key downstream) instead of pooling
        # all FM series into one file distinguished only by an FM column.
        for fm in sorted(df_data.FM.unique()):
            df_fm = df_data[df_data.FM == fm][['P','Exp','W','M']]
            df_fm.to_csv(os.path.join(self.Data_dir, 'generated_Synthetic_exp_data_{}.csv'.format(fm)), index=False)

            tmp_df = []
            for p, exp, tmp in self._iter_p_exp_groups(df_fm):
                W, M = tmp.W.to_numpy(), tmp.M.to_numpy()
                # number of windows is however many full (4 lag + 1 target)
                # points this (P,Exp) group actually has, not a fixed count
                n_windows = len(tmp) - (self.WINDOW - 1)
                for i in range(max(n_windows, 0)):
                    tmp_df.append([p,W[i],M[i],W[i+1],M[i+1],W[i+2],M[i+2],W[i+3],M[i+3],W[i+4],M[i+4]])

            df_train = pd.DataFrame(tmp_df,columns=['P','w1','m1','w2','m2','w3','m3','w4','m4','w5','target'])
            del [tmp_df]
            df_train.P = df_train.P.astype(int)
            df_train.to_csv(os.path.join(self.Data_dir, 'Synthetic_train_data_without_exp_{}.csv'.format(fm)), index=False)

            tmp_df = []
            for p, exp, tmp in self._iter_p_exp_groups(df_fm):
                W, M = tmp.W.to_numpy(), tmp.M.to_numpy()
                n_windows = len(tmp) - (self.WINDOW - 1)
                for i in range(max(n_windows, 0)):
                    tmp_df.append([p,exp,W[i],M[i],W[i+1],M[i+1],W[i+2],M[i+2],W[i+3],M[i+3],W[i+4],M[i+4]])

            df_train = pd.DataFrame(tmp_df,columns=['P','Exp','w1','m1','w2','m2','w3','m3','w4','m4','w5','target'])
            del [tmp_df]
            df_train.P = df_train.P.astype(int)
            df_train.Exp = df_train.Exp.astype(int)
            df_train.to_csv(os.path.join(self.Data_dir, 'Synthetic_train_data_with_exp_{}.csv'.format(fm)), index=False)

            print('FM {} done.'.format(fm))

if __name__=="__main__":
    DATA_DIR = '../Data/2026'
    tr = Prepare_train_data(Data_dir=DATA_DIR)
    tr.train_data()
