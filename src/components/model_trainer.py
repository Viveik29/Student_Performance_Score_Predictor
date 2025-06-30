import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.Logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path1=os.path.join("artifacts","model1.pkl")
    trained_model_file_path2=os.path.join("artifacts","model2.pkl")
    trained_model_file_path3=os.path.join("artifacts","model3.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array1,train_array2,train_array3,test_array1,test_array2,test_array3):
        try:
            logging.info("Split training and test input data")
            X_train1,y_train1,X_train2,y_train2,X_train3,y_train3,X_test1,y_test1,X_test2,y_test2,X_test3,y_test3=(
                train_array1[:,:-1],
                train_array1[:,-1],
                train_array2[:,:-1],
                train_array2[:,-1],
                train_array3[:,:-1],
                train_array3[:,-1],
                test_array1[:,:-1],
                test_array1[:,-1],
                test_array2[:,:-1],
                test_array2[:,-1],
                test_array3[:,:-1],
                test_array3[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            logging.info("Start model evaluation")

            model_report1:dict=evaluate_models(X_train=X_train1,y_train=y_train1,X_test=X_test1,y_test=y_test1,
                                             models=models,param=params)
            model_report2:dict=evaluate_models(X_train=X_train2,y_train=y_train2,X_test=X_test2,y_test=y_test2,
                                             models=models,param=params)
            model_report3:dict=evaluate_models(X_train=X_train3,y_train=y_train3,X_test=X_test3,y_test=y_test3,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score1 = max(sorted(model_report1.values()))
            best_model_score2 = max(sorted(model_report2.values()))
            best_model_score3 = max(sorted(model_report3.values()))

            ## To get best model name from dict

            best_model_name1 = list(model_report1.keys())[
                list(model_report1.values()).index(best_model_score1)
            ]
            best_model1 = models[best_model_name1]

            best_model_name2 = list(model_report2.keys())[
                list(model_report2.values()).index(best_model_score2)
            ]
            best_model2 = models[best_model_name2]

            best_model_name3 = list(model_report3.keys())[
                list(model_report3.values()).index(best_model_score3)
            ]
            best_model3 = models[best_model_name3]

            if best_model_score1<0.6:
                raise CustomException("No best model found")
            if best_model_score2<0.6:
                raise CustomException("No best model found")
            if best_model_score3<0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path1,
                obj=best_model1
            )
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path2,
                obj=best_model2
            )
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path3,
                obj=best_model3
            )

            predicted1=best_model1.predict(X_test1)
            predicted2=best_model2.predict(X_test2)
            predicted3=best_model3.predict(X_test3)

            r2_square1 = r2_score(y_test1, predicted1)
            return r2_square1

            r2_square2 = r2_score(y_test2, predicted2)
            return r2_square2
            
            r2_square3 = r2_score(y_test3, predicted3)
            return r2_square3
            
            
            
        except Exception as e:
            raise CustomException(e,sys)