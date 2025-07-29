from flask import Flask,request,render_template,jsonify
import numpy as np
import pandas as pd
import sys
from src.Logger import logging

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

logging.info(f'Start predicting on input data')
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')

    if request.method == 'POST':
        #  JSON API Request (Postman, frontend)
        if request.is_json:
            try:
                data1 = request.get_json()   

                # Extract fields from JSON
                input_data = CustomData(
                    gender=data1.get('gender'),
                    race_ethnicity=data1.get('race_ethnicity'),
                    parental_level_of_education=data1.get('parental_level_of_education'),
                    lunch=data1.get('lunch'),
                    test_preparation_course=data1.get('test_preparation_course'),
                    reading_score=float(data1.get('writing_score')),
                    writing_score=float(data1.get('reading_score'))
                )

                pred_df = input_data.get_data_as_data_frame()
                predict_pipeline = PredictPipeline()
                results = predict_pipeline.predict(pred_df)

                return jsonify({
                    'prediction': results[0]  # assuming it's a single output
                }), 200

            except Exception as e:
                print("Error:", e, file=sys.stderr)
                return jsonify({'error': str(e)}), 500

    # HTML Form Submission
        else:
            
            try:
                #input_combo = data['input_combo']
                data=CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                writing_score=(request.form.get('writing_score')),
                reading_score=(request.form.get('reading_score')),
                math_score =(request.form.get('math_score')),
                #input_combo = request.form.get("input_combo")
    
            )
                logging.info(f'Data frame created ')
                pred_df = data.get_data_as_data_frame()
                #print(pred_df)
                predict_pipeline = PredictPipeline()
                logging.info(f'As per predicting combo start reading options ')

                input_combo = request.form.get("input_combo")
                logging.info(f'input combo read successfully {input_combo} ')

                if input_combo == "reading_writing":
                    # use math_score and reading_score to predict writing_score
                    pred_df['writing_score'] = float(request.form.get("writing_score"))
                    pred_df['reading_score'] = float(request.form.get("reading_score"))
                    #results = predict_pipeline.predict(pred_df,input_combo)
                    logging.info(f'selected read and write option ')    

                elif input_combo == "math_reading":
                    pred_df['math_score'] = float(request.form.get("math_score"))
                    pred_df['reading_score'] = float(request.form.get("reading_score"))
                    #results = predict_pipeline.predict(pred_df,input_combo )
                    logging.info(f'selected read and math option ')
                    
                elif input_combo == "math_writing":
                    pred_df['math_score'] = float(request.form.get("math_score"))
                    pred_df['writing_score'] = float(request.form.get("writing_score"))
                    #results = predict_pipeline.predict(pred_df,input_combo )
                    logging.info(f'selected math and write option ')

                #logging.info(f'prediticting now ')
                logging.info('Predicting now...')
                results = predict_pipeline.predict(pred_df, input_combo)
                return render_template('home.html', results=results[0])
                                


            except Exception as e:
                print("Error:", e, file=sys.stderr)
                return render_template('home.html', results=f"Error: {e}")

            


if __name__=="__main__":
    app.run(host="0.0.0.0",port=80)   