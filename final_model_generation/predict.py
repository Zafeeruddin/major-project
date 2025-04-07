import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from sklearn.preprocessing import MinMaxScaler

file_path = r"/home/baseer-tuf/zafeer/major/rush-hour-analysis/final_model_generation/footfall_data (4).csv"

def load_models():
    lstm = load_model('/home/baseer-tuf/zafeer/major/rush-hour-analysis/final_model_generation/lstm_model.h5')
    gru = load_model('/home/baseer-tuf/zafeer/major/rush-hour-analysis/final_model_generation/gru_model.h5')
    xgb = joblib.load('/home/baseer-tuf/zafeer/major/rush-hour-analysis/final_model_generation/xgb_model.pkl')
    return lstm, gru, xgb

def load_data(path):
    df = pd.read_csv(path, parse_dates=['datetime'], dayfirst=True)
    df.set_index('datetime', inplace=True)
    df = df.resample('H').mean()  # Ensure hourly frequency
    df['footfall'].interpolate(inplace=True)
    return df

lstm, gru, xgb = load_models()
df = load_data(file_path)
scaler = MinMaxScaler()
values = df['footfall'].values.reshape(-1, 1)
scaled = scaler.fit_transform(values)

def predict_ensemble_for_datetime(dt_str, n_steps=24):
    target = pd.to_datetime(dt_str, dayfirst=True)
    last_timestamp = df.index[-1]
    hours_ahead = int((target - last_timestamp) / pd.Timedelta(hours=1))

    # If the requested datetime is already in the dataset
    if hours_ahead <= 0:
        return float(df.loc[target, 'footfall'])

    # Start from the last 24-hour sequence (scaled)
    seq = scaler.transform(df['footfall'].iloc[-n_steps:].values.reshape(-1, 1)).flatten()

    for _ in range(hours_ahead):
        inp_rnn = seq[-n_steps:].reshape(1, n_steps, 1)
        inp_xgb = seq[-n_steps:].reshape(1, n_steps)

        p_lstm = lstm.predict(inp_rnn)
        p_gru  = gru.predict(inp_rnn)
        p_xgb  = xgb.predict(inp_xgb).reshape(1, 1)

        next_scaled = (p_lstm + p_gru + p_xgb) / 3
        seq = np.append(seq, next_scaled.flatten()[0])

    return float(scaler.inverse_transform([[seq[-1]]])[0][0])

if __name__=="__main__":
    print(predict_ensemble_for_datetime("01-04-2025 17:40"))