import xgboost as xgb
from sklearn import tree
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


from logger.logger import Logger

class MLPredictor():

    def __init__(self, logger:Logger):
        self.logger = logger
        self.logPrefix = "MLModelPredictor"
    
    def run_model(self, X_train, y_train, X_test, y_test, metric_name = "mae", model_name = "linear"):

        self.logger.info(self.logPrefix, "Initializing ML Model Cycle.")

        self.get_ml_model(model_name)

        self.train_model(X_train, y_train)

        y_predict = self.predict(X_test)

        evaluation_score = self.evaluate_model(y_test, y_predict, metric_name)

        return evaluation_score

    def get_ml_model(self, model_name = "linear"):
        
        if model_name == "linear":
            self.logger.info(self.logPrefix, "Initializing Linear Regression model.")
            self.model = LinearRegression()
        
        elif model_name == "ridge":
            self.logger.info(self.logPrefix, "Initializing Ridge Regression model.")
            self.model = Ridge()

        elif model_name == "sgd":
            self.logger.info(self.logPrefix, "Initializing SGD Regression model.")
            self.model = make_pipeline(StandardScaler(),SGDRegressor())

        elif model_name == "svr":
            self.logger.info(self.logPrefix, "Initializing SVR model.")
            self.model = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))

        elif model_name == "tree":
            self.logger.info(self.logPrefix, "Initializing Decision Tree model.")
            self.model = tree.DecisionTreeRegressor()

        elif model_name == "forest":
            self.logger.info(self.logPrefix, "Initializing Random Forest model.")
            self.model = RandomForestRegressor()

        elif model_name == "xgb":
            self.logger.info(self.logPrefix, "Initializing XGBoost model.")
            self.model = xgb.XGBRegressor()

        elif model_name == "dummy_mean":
            self.logger.info(self.logPrefix, "Initializing Dummy Mean model.")
            self.model = DummyRegressor(strategy="mean")

        elif model_name == "dummy_median":
            self.logger.info(self.logPrefix, "Initializing Dummy Median model.")
            self.model = DummyRegressor(strategy="median")

        else:
            self.logger.error(self.logPrefix, "Invalid model name. Ending program.")
            print("Provided model name is invalid.")
            exit()
    
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

            evaluation_score = abs(y_test - y_predict) / len(y_test)
            self.logger.info(self.logPrefix, "MAE score is: " + str(evaluation_score))

            return evaluation_score[0]
        
        elif metric_name == "r2":

            self.logger.info(self.logPrefix, "Using R2 metric.")

            if len(y_test) < 2:
                self.logger.error(self.logPrefix, "Not enough data to calculate R2 metric.")
                print("Not enough data to calculate R2 metric.")
                exit()

            evaluation_score = r2_score(y_test, y_predict)
            self.logger.info(self.logPrefix, "R2 score is: " + str(evaluation_score))

            return evaluation_score
        
        elif metric_name == "mse":

            self.logger.info(self.logPrefix, "Using MSE metric.")

            evaluation_score = mean_squared_error(y_test, y_predict)
            self.logger.info(self.logPrefix, "MSE score is: " + str(evaluation_score))

            return evaluation_score

        else:
            self.logger.error(self.logPrefix, "Invalid metric name. Ending program.")
            print("Provided metric name is invalid.")
            exit()