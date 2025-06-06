import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error
from sklearn import tree
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, SGDRegressor, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
import xgboost as xgb

# Logger class stub (replace with actual implementation if needed)
class Logger:
    def __init__(self, name, level=logging.DEBUG, output="filestream", fileName=None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        fh = logging.FileHandler(fileName) if output == "filestream" else logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def info(self, prefix, message):
        self.logger.info(f"[{prefix}] {message}")

    def error(self, prefix, message):
        self.logger.error(f"[{prefix}] {message}")

class DataHandler():
    def __init__(self, logger:Logger):
        self.logger = logger
        self.logPrefix = "DataHandler"

    def read_manual_data(self):
        self.logger.info(self.logPrefix, "Reading manual data.")
        data_1 = {  # mass and thickness over time
            "hour":      [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 481, 504, 528],
            "mass":      [121.5, 124.18, 126.86, 128.1, 129.35, 130.59, 131.84, 132.66, 134.01, 134.27, 134.48, 135.2, 135.92, 136.64, 135.61, 135.02, 135.29, 136.3, 136.95, 137.61, 138.29, 138.75, 139.05],
            "thickness": [3.45, 3.52, 3.58, 3.6, 3.63, 3.72, 3.75, 3.82, 3.95, 4.03, 4.05, 4.06, 4.06, 4.06, 4.08, 4.08, 4.1, 4.07, 4.073, 4.077, 4.08, 4.09, 4.07]
        }
        data_2 = {  # elasticity e1 over time
            "hour": [0, 24, 90, 180, 270, 360, 528],
            "e1":   [6.479, 4.219, 3.269, 2.757, 2.773536366, 2.185, 2.187]
        }
        return pd.DataFrame(data_1), pd.DataFrame(data_2)

    def get_train_test(self, data, train_index, test_index, target_name, target_columns, normalize_target=False):
        self.logger.info(self.logPrefix, "Splitting train/test data.")
        X_train = data.drop(target_columns, axis=1).iloc[train_index]
        X_test  = data.drop(target_columns, axis=1).iloc[test_index]
        y_train = data[target_name].iloc[train_index]
        y_test  = data[target_name].iloc[test_index]

        X_train = X_train['hour'].to_numpy().reshape(-1, 1)
        y_train = y_train.to_numpy().reshape(-1, 1).ravel()
        X_test  = X_test['hour'].to_numpy().reshape(-1, 1)
        y_test  = y_test.to_numpy().reshape(-1, 1).ravel()

        if normalize_target:
            self.y_mean = y_train.mean()
            self.y_std = y_train.std()
            y_train = (y_train - self.y_mean) / self.y_std
            y_test  = (y_test  - self.y_mean) / self.y_std

        return X_train, y_train, X_test, y_test

class MLPredictor():
    def __init__(self, logger:Logger):
        self.logger = logger
        self.logPrefix = "MLModelPredictor"

    def run_model(self, X_train, y_train, X_test, y_test, model_name="linear"):
        self.logger.info(self.logPrefix, "Running model: " + model_name)
        self.get_ml_model(model_name)
        self.train_model(X_train, y_train)
        y_predict = self.predict(X_test)
        return mean_absolute_error(y_test, y_predict)

    def get_ml_model(self, model_name):
        models = {
            "linear":      LinearRegression(),
            "ridge":       Ridge(),
            "sgd":         make_pipeline(StandardScaler(), SGDRegressor()),
            "svr":         make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2)),
            "tree":        tree.DecisionTreeRegressor(),
            "forest":      RandomForestRegressor(),
            "xgb":         xgb.XGBRegressor(),
            "dummy_mean":  DummyRegressor(strategy="mean"),
            "dummy_median":DummyRegressor(strategy="median")
        }
        if model_name in models:
            self.model = models[model_name]
        else:
            self.logger.error(self.logPrefix, "Invalid model name: " + model_name)
            exit()

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', default='mass', help='Target variable: one of [mass, thickness, e1]')
    parsed_args = parser.parse_args()

    target_name = parsed_args.target

    logger = Logger(name='estimator_logger', level=logging.DEBUG, output="filestream",
                    fileName=datetime.now().strftime('logfile_%d-%m-%Y-%H-%M-%S.%f')[:-3] + ".log")

    data_handler = DataHandler(logger)
    mass_thick, elasticity = data_handler.read_manual_data()

    # Choose dataset and target columns based on user selection
    if target_name == "e1":
        data = elasticity
        target_columns = ["e1"]
    else:
        data = mass_thick
        target_columns = ["mass", "thickness"]

    loo = LeaveOneOut()
    models = ["linear", "ridge", "sgd", "svr", "tree", "forest", "xgb", "dummy_mean", "dummy_median"]
    model_scores = {}

    for model_name in models:
        ml_pred = MLPredictor(logger)
        total_score = 0
        num_of_splits = 0
        for train_idx, test_idx in loo.split(data):
            norm_target = model_name in ["svr", "sgd"]
            X_train, y_train, X_test, y_test = data_handler.get_train_test(
                data, train_idx, test_idx, target_name, target_columns, normalize_target=norm_target
            )
            score = ml_pred.run_model(X_train, y_train, X_test, y_test, model_name=model_name)
            total_score += score
            num_of_splits += 1
        model_scores[model_name] = total_score / num_of_splits

    for model_name, score in model_scores.items():
        print(f'MAE for {model_name}: {round(score, 3)}')

    plt.figure(figsize=(10, 5))
    plt.bar(model_scores.keys(), model_scores.values())
    plt.ylabel('Mean Absolute Error')
    plt.title(f'Model Comparison (MAE) for target: {target_name}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
