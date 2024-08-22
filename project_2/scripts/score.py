import json
import numpy as np
import joblib

def init():
    global model
    model_path = 'models/diabetes_model.pkl'
    model = joblib.load(model_path)

def run(raw_data):
    data = json.loads(raw_data)['data']
    prediction = model.predict(np.array(data))
    return prediction.tolist()
