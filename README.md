Student Performance Score Predictor:

This project is a Flask-based web application that predicts a student’s performance in one subject based on their scores in the other two subjects (Math, Reading, Writing). It supports both web-based form input and JSON API-based requests.
________________________________________
✨ Key Features
•	Predict student scores based on two subject inputs
•	Supports three combinations:
o	Predict Math using Reading + Writing
o	Predict Reading using Math + Writing
o	Predict Writing using Math + Reading
•	Web form UI + JSON API endpoint
•	CustomData class to structure input
•	Modular pipeline structure for maintainability
________________________________________
📂 Project Structure
student_score_predictor/
|— app.py                    # Main Flask application
|— templates/
|   |— home.html             # HTML form for input and output
|— src/
    |— pipeline/
        |— predict_pipeline.py  # Contains CustomData and PredictPipeline classes
    |— Logger.py             # Optional logging module
|— README.md                # This file
________________________________________
⚡ Setup Instructions
1. Clone the Repository
2. https://github.com/Viveik29/Student_Performance_Score_Predictor.git
cd student_score_predictor
3. Create a Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
4. Install Required Libraries
pip install -r requirements.txt
5. Run the App
python app.py
Open your browser and navigate to http://localhost:80 or http://127.0.0.1
________________________________________
🌐 API Usage (POST JSON)
Endpoint
POST /predictdata
Content-Type: application/json
JSON Input Examples
Predict Math:
{
  "input_combo": "reading_writing",
  "reading_score": 78,
  "writing_score": 82
}
Predict Writing:
{
  "input_combo": "math_reading",
  "math_score": 75,
  "reading_score": 88
}
Predict Reading:
{
  "input_combo": "math_writing",
  "math_score": 90,
  "writing_score": 84
}
JSON Output
{
  "prediction": 83.5
}
________________________________________
📅 Input Modes Supported
•	HTML Form Input using dropdown and numeric input
•	Postman/Fetch/XHR style JSON input via REST API
________________________________________
📊 Model Overview
•	Trained using regression models on student performance dataset
•	Targets one subject while using two others as predictors
•	Uses separate pipelines for each prediction mode
________________________________________
________________________________________
📍 Credits
•	Built with Flask, Scikit-learn, Pandas
•	Dataset source: Open-source student performance dataset
_____________________________________________
🌟 Sample Screenshot:
<img width="861" height="585" alt="image" src="https://github.com/user-attachments/assets/16216b6e-e661-4ef8-aec0-fd218049861e" />

____________________________________




🚀 Author
Viveik
ML & MLOps Engineer
