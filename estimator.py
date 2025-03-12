import logging
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.model_selection import LeaveOneOut

import xgboost as xgb
from sklearn.linear_model import LinearRegression



from data_handler  import *
from logger.logger import Logger

if __name__ == '__main__':
    
    logger = Logger(name = 'estimator_logger', level=logging.DEBUG, output="filestream",
                        fileName=datetime.now().strftime('logfile_%d-%m-%Y-%H-%M-%S.%f')[:-3] + ".log")

    log_prefix = "Main Estimator"
    logger.info(log_prefix, "Starting estimator and initializing logger.")

    logger.info(log_prefix, "Reading the data.")
    mass_thick, elasticity = read_manual_data()

    
    logger.info(log_prefix, "Starting Leave one Out Process.")

    total_mae = 0
    num_of_splits = 0

    loo = LeaveOneOut()
    for train_index, test_index in loo.split(mass_thick): 

        
        logger.info(log_prefix, "Splitting train set.")
        X_train, X_test = mass_thick.drop(['mass', 'thickness'], axis=1).iloc[train_index], mass_thick.drop(['mass', 'thickness'], axis=1).iloc[test_index]

        logger.info(log_prefix, "Splitting test set.")
        y_train, y_test = mass_thick['mass'].iloc[train_index], mass_thick['mass'].iloc[test_index]

        # linear_regressor = LinearRegression()
        # linear_regressor.fit(X_train['hour'].to_numpy().reshape(-1,1), y_train.to_numpy().reshape(-1,1))
        # y_predict = linear_regressor.predict(X_test['hour'].to_numpy().reshape(-1,1))

        clf = xgb.XGBRegressor()
        clf.fit(X_train['hour'].to_numpy().reshape(-1,1), y_train.to_numpy().reshape(-1,1))
        y_predict = clf.predict(X_test['hour'].to_numpy().reshape(-1,1))

        curr_mae = abs(y_test.to_numpy().reshape(-1,1) - y_predict)

        total_mae += curr_mae
        num_of_splits += 1

        # plt.scatter(X_train['hour'].to_numpy(), y_train.to_numpy(),color='g')
        # plt.plot(X_test['hour'].to_numpy(), clf.predict(X_test['hour'].to_numpy().reshape(-1,1)),color='k')

        # plt.show()

        print(curr_mae)

    print(f'Mean Absolute Error is {total_mae/num_of_splits}')