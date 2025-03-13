import logging
import argparse
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.model_selection import LeaveOneOut


from data_handler  import *
from ml_models     import MLPredictor
from logger.logger import Logger

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model',      help='Define the type of regressor to use. on of [linear, ridge, sgd, svr, tree, forest, xgb]',  default='linear')
    parser.add_argument('-t', '--target',     help='Define the target variable. on of [mass, thick, all]',                                     default='mass')
    parser.add_argument('-e', '--evaluation', help='Define the evaluation metric.',                                                            default='mae')
    parsed_args = parser.parse_args()

    model_name = parsed_args.model
    target_name = parsed_args.target
    metric_name = parsed_args.evaluation


    logger = Logger(name = 'estimator_logger', level=logging.DEBUG, output="filestream",
                        fileName=datetime.now().strftime('logfile_%d-%m-%Y-%H-%M-%S.%f')[:-3] + ".log")

    log_prefix = "Main Estimator"
    logger.info(log_prefix, "Starting estimator and initializing logger.")

    logger.info(log_prefix, "Reading the data.")
    
    mass_thick, elasticity = read_manual_data()

    logger.info(log_prefix, "Creating Model Predictor.")
    ml_pred = MLPredictor(logger)

    logger.info(log_prefix, "Starting Leave one Out Process.")

    total_mae = 0
    num_of_splits = 0

    loo = LeaveOneOut()
    for train_index, test_index in loo.split(mass_thick): 


        logger.info(log_prefix, "Splitting train set.")
        X_train, X_test = mass_thick.drop(['mass', 'thickness'], axis=1).iloc[train_index], mass_thick.drop(['mass', 'thickness'], axis=1).iloc[test_index]

        logger.info(log_prefix, "Splitting test set.")
        y_train, y_test = mass_thick['mass'].iloc[train_index], mass_thick['mass'].iloc[test_index]


        X_train = X_train['hour'].to_numpy().reshape(-1,1)
        y_train = y_train.to_numpy().reshape(-1,1).ravel()

        X_test = X_test['hour'].to_numpy().reshape(-1,1)
        y_test = y_test.to_numpy().reshape(-1,1).ravel()

        score = ml_pred.run_model(X_train, y_train, X_test, y_test, metric_name, model_name)
        print("MAE is: " + str(score))


        curr_mae = score

        total_mae += curr_mae
        num_of_splits += 1

    logger.info(log_prefix, "Mean Absolute Error over all splits:" + str(round(total_mae/num_of_splits,3)))
    print(f'Mean Absolute Error is {round(total_mae/num_of_splits,3)}')