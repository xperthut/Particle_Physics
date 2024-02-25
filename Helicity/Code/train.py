from model import Model
import joblib
import pandas as pd
import numpy as np
import argparse
import os

class Train:
    def __init__(self, name):
        self.name = name
        self.model = Model()
        
    def train_model(self, df_train, with_exp_id='No'):
        save_model_name = f"{self.name}_exp_{with_exp_id}"
        print(save_model_name)
        
        if self.name=='GB':
            model = self.model.GradientBoostedModel(df_train, 1)
        elif self.name=='RF':
            model = self.model.RandomForestModel(df_train, 1)
        elif self.name=='XGB':
            model = self.model.XGBoostModel(df_train, 1)
            
        joblib.dump(model, f'../model/{save_model_name}.sav')
        print('Done training and saved')
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train_data", type=str, required=True, help="Path of train data with file name")
    parser.add_argument("--model_type", type=str, required=True, help="GB for Gradient boosted, RF for Random forest, and XGB for Extreme gradient boosted.")
    parser.add_argument("--with_exp_id", type=str, required=False, default="No", help="Saved model name")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.train_data):
        print(f"{args.train_data} doen't exists")
    elif args.model_type not in ['GB', 'XGB', 'RF']:
        print("Invalid model type.")
    else:
        df_train = pd.read_csv(args.train_data)
        tr = Train(args.model_type)
        tr.train_model(df_train, args.with_exp_id)
        
    