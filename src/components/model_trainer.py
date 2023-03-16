import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.utils import save_obj, evaluate_models
from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting train and test data.')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1], train_array[:,-1],
                test_array[:,:-1], test_array[:,-1]
            )

            models = {
                'LinearRegression': LinearRegression(),
                'SVM': SVR(),
                'KNeigbors': KNeighborsRegressor(),
                'DecisionTree': DecisionTreeRegressor(),
                'RandomForest': RandomForestRegressor(),
                'AdaBoost': AdaBoostRegressor(),
                'GradientBoost': GradientBoostingRegressor(),
                'CatBoost': CatBoostRegressor(),
                'XGBoost': XGBRegressor()
            }

            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
        
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No best model found')
            logging.info('Found best model on both train and test data.')

            save_obj(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            model_r2_score = r2_score(y_test, predicted)
            return model_r2_score

        except Exception as e:
            raise CustomException(e, sys)