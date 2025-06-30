import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.Logger import logging
import os


class PredictPipeline:
    def __init__(self):
       
        pass

    def predict(self,features,input_combo):
        try: 
            model_path1=os.path.join("artifacts","model1.pkl")
            model_path2=os.path.join("artifacts","model2.pkl")
            model_path3=os.path.join("artifacts","model3.pkl")
            preprocessor_path1=os.path.join('artifacts','preprocessor1.pkl')
            preprocessor_path2=os.path.join('artifacts','preprocessor2.pkl')
            preprocessor_path3=os.path.join('artifacts','preprocessor3.pkl')
            print("Before Loading")
            model1=load_object(file_path=model_path1)
            model2=load_object(file_path=model_path2)
            model3=load_object(file_path=model_path3)
            if input_combo == "reading_writing":
                preprocessor1=load_object(file_path=preprocessor_path1)
                logging.info(f'Started loading preprocessor file')
                print("Preprocessor type:", type(preprocessor1))
                data_scaled=preprocessor1.transform(features)
                logging.info(f'Successfully loaded and transformed preprocessor file')
                preds=model1.predict(data_scaled)
                logging.info(f'Successfully loaded Model1 file')
                return preds
            elif input_combo == "math_reading":
                preprocessor2=load_object(file_path=preprocessor_path2)
                data_scaled=preprocessor2.transform(features)
                preds=model2.predict(data_scaled)
                return preds
            elif input_combo == 'math_writing':
                preprocessor3=load_object(file_path=preprocessor_path3)
                data_scaled=preprocessor3.transform(features)
                preds=model3.predict(data_scaled)
                return preds
            else:
                print("Invalid input_combo value:", input_combo)
                return ["Invalid input_combo"]



            #preprocessor1=load_object(file_path=preprocessor_path1)
            #preprocessor2=load_object(file_path=preprocessor_path2)
            #preprocessor3=load_object(file_path=preprocessor_path3)
            print("After Loading")
            #data_scaled=preprocessor1.transform(features)
            #data_scaled=preprocessor2.transform(features)
            #data_scaled=preprocessor3.transform(features)
            # preds=model1.predict(data_scaled)
            # preds=model2.predict(data_scaled)
            # preds=model3.predict(data_scaled)
            #return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
        math_score:int
       #input_combo 
        # reading_writing : None,
        # math_writing : None,
        # math_reading : None



        ):

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

        self.math_score = math_score

        #self.input_combo = input_combo
        # self.reading_writing = reading_writing
        # self.math_writing = math_writing
        # self.math_reading = math_reading

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
                "math_score" : [self.math_score],
                #"input_combo" :[self.input_combo]
            }
            


            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

