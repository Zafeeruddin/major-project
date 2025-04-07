import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.losses import MeanSquaredError
from xgboost import XGBRegressor
import joblib

# Custom LSTM layer to handle deprecated 'time_major'
class CustomLSTM(LSTM):
    def __init__(self, units, **kwargs):
        kwargs.pop('time_major', None)  # Remove 'time_major' if present
        super().__init__(units, **kwargs)

def predict_footfall(datetime_str):
    # Load scaler and models with custom objects
    scaler = joblib.load('models/scaler.pkl')
    
    # Load LSTM model
    lstm_model = load_model(
        'models/lstm_model.h5',
        custom_objects={'LSTM': CustomLSTM, 'mse': MeanSquaredError()}
    )
    
    # Load GRU model
    gru_model = load_model(
        'models/gru_model.h5',
        custom_objects={'LSTM': CustomLSTM, 'mse': MeanSquaredError()}
    )
    
    # Load XGBoost model using native method
    xgb_model = XGBRegressor()
    xgb_model.load_model('models/xgb_model.json')  # Replace .json with your format
    
    # ... (rest of your prediction code)

if __name__ == '__main__':
    date_input = input("Enter the datetime (DD-MM-YYYY HH:MM): ")
    prediction = predict_footfall(date_input)
    print(f"Predicted footfall for {date_input} is: {prediction:.2f}")