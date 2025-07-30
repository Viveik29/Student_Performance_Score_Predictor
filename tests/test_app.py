import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from app import app

@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client

def test_get_predict_form(client):
    """Test GET request for loading the home form"""
    response = client.get('/predictdata')
    assert response.status_code == 200
    assert b"<form" in response.data  # Basic check for form existence

# --- POST Tests via JSON API ---

def test_post_predict_reading_writing(client):
    """Test reading_writing combo (predict writing_score)"""
    payload = {
        "gender": "female",
        "race_ethnicity": "group B",
        "parental_level_of_education": "associate's degree",
        "lunch": "standard",
        "test_preparation_course": "completed",
        "math_score": 75,
        "reading_score": 85,
        "writing_score": 0,  # placeholder, not used in prediction
        "input_combo": "reading_writing"
    }

    response = client.post('/predictdata', data=payload, content_type='application/x-www-form-urlencoded')
    assert response.status_code == 200
    assert b"prediction" in response.data or b"result" in response.data

def test_post_predict_math_reading(client):
    """Test math_reading combo (predict reading_score)"""
    payload = {
        "gender": "male",
        "race_ethnicity": "group C",
        "parental_level_of_education": "master's degree",
        "lunch": "free/reduced",
        "test_preparation_course": "none",
        "math_score": 82,
        "writing_score": 88,
        "reading_score": 0,
        "input_combo": "math_reading"
    }

    response = client.post('/predictdata', data=payload, content_type='application/x-www-form-urlencoded')
    assert response.status_code == 200

def test_post_predict_math_writing(client):
    """Test math_writing combo (predict reading_score)"""
    payload = {
        "gender": "female",
        "race_ethnicity": "group A",
        "parental_level_of_education": "high school",
        "lunch": "standard",
        "test_preparation_course": "completed",
        "math_score": 90,
        "reading_score": 0,
        "writing_score": 84,
        "input_combo": "math_writing"
    }

    response = client.post('/predictdata', data=payload, content_type='application/x-www-form-urlencoded')
    assert response.status_code == 200
