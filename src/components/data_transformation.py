import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.Logger import logging
import os

from src.utils import save_object

@dataclass 
class DataTransformationConfig:
    preprocessor_obj_file_path1=os.path.join('artifacts',"proprocessor1.pkl")
    preprocessor_obj_file_path2=os.path.join('artifacts',"proprocessor2.pkl")
    preprocessor_obj_file_path3=os.path.join('artifacts',"proprocessor3.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self,DF):
        '''
        This function is responsible for data trnasformation
        
        '''
        try:
            # define numerical & categorical columns
            #df=pd.read_csv(r"notebook\data\stud.csv")
            numerical_columns = [feature for feature in DF.columns if DF[feature].dtype != 'O']
            categorical_columns = [feature for feature in DF.columns if DF[feature].dtype == 'O']
            # numerical_columns = ["writing_score", "reading_score"]
            # categorical_columns = [
            #     "gender",
            #     "race_ethnicity",
            #     "parental_level_of_education",
            #     "lunch",
            #     "test_preparation_course",
            # ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor.fit_transform(DF)
        
        except Exception as e:
            raise CustomException(e,sys)
    def target_column_selection(self):
        target_column_name1="math_score"
        target_column_name2="writing_score"
        target_column_name3="reading_score" 
        
        return target_column_name1,target_column_name2,target_column_name3


        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            target_col1,target_col2,target_col3 = self.target_column_selection()
            logging.info("Target column selected")

            # preprocessing_obj1=self.get_data_transformer_object(x)
            # preprocessing_obj2=self.get_data_transformer_object(y)
            # preprocessing_obj3=self.get_data_transformer_object(z)

            #target_column_name="math_score"
            #numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df1=train_df.drop(columns=[target_col1],axis=1)
            input_feature_train_df2=train_df.drop(columns=[target_col2],axis=1)
            input_feature_train_df3=train_df.drop(columns=[target_col3],axis=1)
            logging.info("Input features of train_data selected ")

            target_feature_train_df1=train_df[target_col1]
            target_feature_train_df2=train_df[target_col2]
            target_feature_train_df3=train_df[target_col3]
            logging.info("Target features of train_data selected ")




            input_feature_test_df1=test_df.drop(columns=[target_col1],axis=1)
            input_feature_test_df2=test_df.drop(columns=[target_col2],axis=1)
            input_feature_test_df3=test_df.drop(columns=[target_col3],axis=1)


            target_feature_test_df1=test_df[target_col1]
            target_feature_test_df2=test_df[target_col2]
            target_feature_test_df3=test_df[target_col3]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            # preprocessing_obj1=self.get_data_transformer_object(x)
            # preprocessing_obj2=self.get_data_transformer_object(y)
            # preprocessing_obj3=self.get_data_transformer_object(z)

            input_feature_train_arr1=self.get_data_transformer_object(input_feature_train_df1)
            input_feature_train_arr2=self.get_data_transformer_object(input_feature_train_df2)
            input_feature_train_arr3=self.get_data_transformer_object(input_feature_train_df3)
            #input_feature_train_arr1 = preprocessor.fit_transform(input_feature_train_df1)
            logging.info("function testing")
            # input_feature_train_arr1=preprocessing_obj1.fit_transform(input_feature_train_df1)
            # input_feature_train_arr2=preprocessing_obj2.fit_transform(input_feature_train_df2)
            # input_feature_train_arr3=preprocessing_obj3.fit_transform(input_feature_train_df3)


            # input_feature_test_arr1=preprocessing_obj1.transform(input_feature_test_df1)
            # input_feature_test_arr2=preprocessing_obj2.transform(input_feature_test_df2)
            # input_feature_test_arr3=preprocessing_obj3.transform(input_feature_test_df3)
            input_feature_test_arr1=self.get_data_transformer_object(input_feature_test_df1)
            input_feature_test_arr2=self.get_data_transformer_object(input_feature_test_df2)
            input_feature_test_arr3=self.get_data_transformer_object(input_feature_test_df3)


            train_arr1 = np.c_[input_feature_train_arr1, np.array(target_feature_train_df1)]
            train_arr2 = np.c_[input_feature_train_arr2, np.array(target_feature_train_df2)]
            train_arr3 = np.c_[input_feature_train_arr3, np.array(target_feature_train_df3)]

            logging.info("Trainning array created")


            test_arr1 = np.c_[input_feature_test_arr1, np.array(target_feature_test_df1)]
            test_arr2 = np.c_[input_feature_test_arr2, np.array(target_feature_test_df2)]
            test_arr3 = np.c_[input_feature_test_arr3, np.array(target_feature_test_df3)]

            logging.info("Testing array created")

            logging.info(f"Saved preprocessing object.")

            save_object(
            
                file_path=self.data_transformation_config.preprocessor_obj_file_path1,
                obj=input_feature_train_arr1

            )
            save_object(
              
                file_path=self.data_transformation_config.preprocessor_obj_file_path2,
                obj=input_feature_train_arr2

            )
            save_object(
             
                file_path=self.data_transformation_config.preprocessor_obj_file_path3,
                obj=input_feature_train_arr3

            )

            return (
                train_arr1,
                train_arr2,
                train_arr3,
                test_arr1,
                test_arr2,
                test_arr3,
                self.data_transformation_config.preprocessor_obj_file_path1,
                self.data_transformation_config.preprocessor_obj_file_path2,
                self.data_transformation_config.preprocessor_obj_file_path3
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    DataTransformation()

