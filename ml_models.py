import xgboost as xgb
from sklearn.linear_model import LinearRegression

from logger.logger import Logger

class MLPredictor():

    def __init__(self, logger:Logger):
        self.logger = logger
        self.logPrefix = "MLModelPredictor"
    
    def run_model(self, X_train, y_train, X_test, y_test, metric_name = "mae", model_name = "lr", ):

        self.logger.info(self.logPrefix, "Initializing ML Model Cycle.")

        self.get_ml_model(model_name)

        self.train_model(X_train, y_train)

        y_predict = self.predict(X_test)

        evaluation_score = self.evaluate_model(y_test, y_predict, metric_name)

        return evaluation_score

    def get_ml_model(self, model_name = "lr"):
        
        if model_name == "lr":
            self.logger.info(self.logPrefix, "Initializing Linear Regression model.")
            self.model = LinearRegression()
        elif model_name == "xgb":
            self.logger.info(self.logPrefix, "Initializing XGBoost model.")
            self.model = xgb.XGBRegressor()
        else:
            self.logger.error(self.logPrefix, "Invalid model name.")
            print("Provided model name is invalid.")
            return
    
    def train_model(self, X_train, y_train):
        
        self.logger.info(self.logPrefix, "Training model with provided data.")
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        
        self.logger.info(self.logPrefix, "Getting model predictions.")
        return self.model.predict(X_test)

    def evaluate_model(self, y_test, y_predict, metric_name = "mae"):

        self.logger.info(self.logPrefix, "Getting evaluation metric.")

        if metric_name == "mae":

            self.logger.info(self.logPrefix, "Using MAE metric.")

            evaluation_score = abs(y_test - y_predict)
            self.logger.info(self.logPrefix, "MAE score is: " + str(evaluation_score))
            print("MAE score is: " + str(evaluation_score[0][0]))

            return evaluation_score
        
        else:
            self.logger.error(self.logPrefix, "Invalid metric name.")
            print("Provided metric name is invalid.")