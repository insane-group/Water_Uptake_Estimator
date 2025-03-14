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
    parser.add_argument('-t', '--target',     help='Define the target variable. on of [mass, thickness, all]',                                 default='mass')
    parser.add_argument('-e', '--evaluation', help='Define the evaluation metric.',                                                            default='mae')
    parsed_args = parser.parse_args()

    model_name  = parsed_args.model
    target_name = parsed_args.target
    metric_name = parsed_args.evaluation


    logger = Logger(name = 'estimator_logger', level=logging.DEBUG, output="filestream",
                        fileName=datetime.now().strftime('logfile_%d-%m-%Y-%H-%M-%S.%f')[:-3] + ".log")

    log_prefix = "Main Estimator"
    logger.info(log_prefix, "Starting estimator and initializing logger.")

    logger.info(log_prefix, "Initializing data handler.")
    data_handler = DataHandler(logger)
    
    logger.info(log_prefix, "Reading the data.")
    
    mass_thick, elasticity = data_handler.read_manual_data()

    logger.info(log_prefix, "Creating Model Predictor.")
    ml_pred = MLPredictor(logger)

    logger.info(log_prefix, "Starting Leave one Out Process.")

    total_score = 0
    total_dummy_mean_score = 0
    total_dummy_median_score = 0
    num_of_splits = 0

    loo = LeaveOneOut()
    
    data = mass_thick
    data_splits = loo.split(mass_thick)
    target_columns = ["mass", "thickness"]
    
    
    if target_name == "e1":
        data = elasticity
        data_splits = loo.split(elasticity)
        target_columns = ["e1"]

    evaluation_metric = "Mean Absolute Error"
    if metric_name == "r2":
        evaluation_metric = "R^2"
    elif metric_name == "mse":
        evaluation_metric = "Mean Squared Error"

    for train_index, test_index in data_splits: 

        X_train, y_train, X_test, y_test = data_handler.get_train_test(data, train_index, test_index, target_name, target_columns)

        logger.info(log_prefix, "Running test model.")
        real_score = ml_pred.run_model(X_train, y_train, X_test, y_test, metric_name, model_name)
        print(f"{evaluation_metric} is: {str(round(real_score,3))}")


        logger.info(log_prefix, "Running dummy models.")
        dummy_mean_score   = ml_pred.run_model(X_train, y_train, X_test, y_test, metric_name, "dummy_mean")
        dummy_median_score = ml_pred.run_model(X_train, y_train, X_test, y_test, metric_name, "dummy_median")

        total_score              += real_score
        total_dummy_mean_score   += dummy_mean_score
        total_dummy_median_score += dummy_median_score

        num_of_splits += 1


    logger.info(log_prefix, evaluation_metric + " over all splits for tested regressor is:" + str(round(total_score/num_of_splits,3)))
    print(f'{evaluation_metric} over all splits for tested regressor is: {round(total_score/num_of_splits,3)}')

    logger.info(log_prefix, evaluation_metric + " over all splits for dummy mean regressor is:" + str(round(total_dummy_mean_score/num_of_splits,3)))
    print(f'{evaluation_metric} over all splits for dummy mean regressor is: {round(total_dummy_mean_score/num_of_splits,3)}')

    logger.info(log_prefix, evaluation_metric + " over all splits for dummy median regressor is:" + str(round(total_dummy_median_score/num_of_splits,3)))
    print(f'{evaluation_metric} over all splits for dummy median regressor is: {round(total_dummy_median_score/num_of_splits,3)}')