import numpy as np
import pandas as pd

from logger.logger import Logger

class DataHandler():

    def __init__(self, logger:Logger):
        
        self.logger = logger
        self.logPrefix = "DataHandler"

    def read_manual_data(self) -> pd.DataFrame:

        self.logger.info(self.logPrefix, "Reading manual data.")

        data_1 = {"hour"      : [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 481, 504, 528],
                  "mass"      : [121.5,	124.18,	126.86,	128.1,	129.35,	130.59,	131.84,	132.66,	134.01,	134.27,	134.48,	135.2,	135.92,	136.64,	135.61,	135.02,	135.29,	136.3,	136.95,	137.61,	138.29,	138.75,	139.05],
                  "thickness" : [3.45,	3.52,	3.58,	3.6,	3.63,	3.72,	3.75,	3.82,	3.95,	4.03,	4.05,	4.06,	4.06,	4.06,	4.08,	4.08,	4.1,	4.07,	4.073,	4.077,	4.08,	4.09,	4.07]
                 }
        
        data_2 = {"hour"      : [0,     24,    90,     180,    270,         360,   528],
                  "e1"        : [6.479, 4.219, 3.269,  2.757,  2.773536366, 2.185, 2.187]
                 }
        
        return pd.DataFrame(data_1), pd.DataFrame(data_2)

    def get_train_test(self, data:pd.DataFrame, train_index, test_index, target_name:str, target_columns:list):
        
        self.logger.info(self.logPrefix, "Splitting train set.")
        X_train, X_test = data.drop(target_columns, axis=1).iloc[train_index], data.drop(target_columns, axis=1).iloc[test_index]

        self.logger.info(self.logPrefix, "Splitting test set.")
        y_train, y_test = data[target_name].iloc[train_index], data[target_name].iloc[test_index]


        X_train = X_train['hour'].to_numpy().reshape(-1,1)
        y_train = y_train.to_numpy().reshape(-1,1).ravel()

        X_test = X_test['hour'].to_numpy().reshape(-1,1)
        y_test = y_test.to_numpy().reshape(-1,1).ravel()

        return X_train, y_train, X_test, y_test

    def get_data_std(self, data:pd.DataFrame, target_name:str):
    
        self.logger.info(self.logPrefix, "Getting standard deviation for " + target_name + ".")
        return round(np.std(data[target_name]),3)