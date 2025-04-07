# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
import tensorflow as tf

import pandas as pd
import matplotlib.pyplot as plt

# Update this path if needed
file_path = r"/home/baseer-tuf/zafeer/major/rush-hour-analysis/final_model_generation/footfall_data (4).csv"

# Load the data, parsing the “datetime” column (day‑first format)
df = pd.read_csv(
    file_path,
    parse_dates=['datetime'],
    dayfirst=True
)

# Plot
plt.figure(figsize=(12, 5))
plt.plot(df['datetime'], df['footfall'])
plt.xlabel('Date & Time')
plt.ylabel('Footfall')
plt.title('Footfall Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt

# file_path = r"C:\Users\zaidb\Downloads\footfall_data (4).csv"

# Load and parse datetime
df = pd.read_csv(file_path, parse_dates=['datetime'], dayfirst=True)

# Extract weekday (0=Mon, …,6=Sun)
# df['day_of_week'] = df['datetime'].dt.dayofweek
df['day_of_week'] = (df['datetime'].dt.dayofweek - 1) % 7
# Compute average footfall per weekday
avg_footfall = df.groupby('day_of_week')['footfall'].mean()

# Plot
plt.figure(figsize=(10,5))
avg_footfall.plot(kind='bar')
plt.xticks(ticks=range(7), labels=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], rotation=45)
plt.ylabel('Average Footfall')
plt.title('Average Footfall by Day of Week')
plt.tight_layout()
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt

# File path
# file_path = r"C:\Users\zaidb\Downloads\footfall_data (4).csv"

# Load & parse datetime
df = pd.read_csv(file_path, parse_dates=['datetime'], dayfirst=True)

# Extract day‑of‑week (0=Mon…6=Sun) and hour
# df['day_of_week'] = df['datetime'].dt.dayofweek
df['day_of_week'] = (df['datetime'].dt.dayofweek - 1) % 7

df['hour'] = df['datetime'].dt.hour

# Filter to hours 5–23
df = df[df['hour'].between(5, 23)]

# Compute average footfall for each (hour, day_of_week)
avg = df.groupby(['hour', 'day_of_week'])['footfall'].mean().unstack()

# Plot one line per weekday
plt.figure(figsize=(12, 6))
weekday_names = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
for dow in range(7):
    plt.plot(avg.index, avg[dow], marker='o', label=weekday_names[dow])

plt.xticks(range(5, 24))
plt.xlabel('Hour of Day')
plt.ylabel('Average Footfall')
plt.title('Average Hourly Footfall by Day of Week (5:00–23:00)')
plt.legend(title='Day of Week')
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt

# file_path = r"/home/baseer-tuf/zafeer/major/rush-hour-analysis/footfall_data (4).csv"

# Load & parse datetime
df = pd.read_csv(file_path, parse_dates=['datetime'], dayfirst=True)

# Extract weekday (0=Mon … 6=Sun) and hour
# df['day_of_week'] = df['datetime'].dt.dayofweek
df['day_of_week'] = (df['datetime'].dt.dayofweek - 1) % 7

df['hour'] = df['datetime'].dt.hour

# Filter to hours 5–23
df = df[df['hour'].between(5, 23)]

# Compute average footfall per hour per weekday
heatmap_data = df.groupby(['hour', 'day_of_week'])['footfall'] \
                 .mean() \
                 .unstack()

# Plot heatmap
fig, ax = plt.subplots(figsize=(10, 12))
im = ax.imshow(heatmap_data.values, aspect='auto', origin='lower')

# Axis ticks
ax.set_xticks(range(7))
ax.set_xticklabels(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], rotation=45)
ax.set_yticks(range(len(heatmap_data.index)))
ax.set_yticklabels(heatmap_data.index)

# Annotate each cell
for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        ax.text(j, i, f"{heatmap_data.iat[i, j]:.0f}", ha='center', va='center', fontsize=6)

ax.set_xlabel('Day of Week')
ax.set_ylabel('Hour of Day')
ax.set_title('Average Footfall Heatmap (Hour vs. Day of Week)')

cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Average Footfall')

plt.tight_layout()
plt.show()


# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.models import load_model
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import joblib
# Load and prepare data
def load_data(path):
    df = pd.read_csv(path, parse_dates=['datetime'], dayfirst=True)
    df.set_index('datetime', inplace=True)
    df = df.resample('H').mean()  # Ensure hourly frequency
    df['footfall'].interpolate(inplace=True)
    return df

# file_path = r"C:\Users\zaidb\Downloads\footfall_data (4).csv"
df = load_data(file_path)

# Scale values
scaler = MinMaxScaler()
values = df['footfall'].values.reshape(-1, 1)
scaled = scaler.fit_transform(values)

# Create supervised dataset (24-hour lookback)
def create_sequences(data, n_steps=24):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

n_steps = 24
X, y = create_sequences(scaled, n_steps)

# Split train/test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape for RNNs
X_train_rnn = X_train.reshape((X_train.shape[0], n_steps, 1))
X_test_rnn = X_test.reshape((X_test.shape[0], n_steps, 1))

# Build models
# def build_lstm():
#     model = Sequential([LSTM(50, input_shape=(n_steps, 1)), Dense(1)])
#     model.compile(optimizer='adam', loss='mse')
#     return model


def load_models():
    lstm = load_model('lstm_model.h5')
    gru = load_model('gru_model.h5')
    xgb = joblib.load('xgb_model.pkl')
    return lstm, gru, xgb


def save_models(lstm, gru, xgb):
    print("\n\n.................Ready to get saved......................................\n\n")
    lstm.save('lstm_model.h5')
    gru.save('gru_model.h5')
    joblib.dump(xgb, 'xgb_model.pkl')


def build_lstm(n_steps):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, input_shape=(n_steps, 1)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_gru():
    model = Sequential([GRU(50, input_shape=(n_steps, 1)), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    return model

# Train LSTM
lstm = build_lstm(n_steps)
lstm.fit(X_train_rnn, y_train, epochs=20, batch_size=32, validation_data=(X_test_rnn, y_test))

# Train GRU
gru = build_gru()
gru.fit(X_train_rnn, y_train, epochs=20, batch_size=32, validation_data=(X_test_rnn, y_test))

# Train XGBoost (flatten inputs)
X_train_xgb = X_train.reshape(X_train.shape[0], -1)
X_test_xgb = X_test.reshape(X_test.shape[0], -1)
xgb = XGBRegressor(n_estimators=100)
xgb.fit(X_train_xgb, y_train)

# Predictions
preds_lstm = scaler.inverse_transform(lstm.predict(X_test_rnn))
preds_gru = scaler.inverse_transform(gru.predict(X_test_rnn))
preds_xgb = scaler.inverse_transform(xgb.predict(X_test_xgb).reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluate
for name, pred in [('LSTM', preds_lstm), ('GRU', preds_gru), ('XGBoost', preds_xgb)]:
    rmse = np.sqrt(mean_squared_error(y_test_inv, pred))
    print(f"{name} RMSE: {rmse:.2f}")

# Plot actual vs predicted
plt.figure(figsize=(12, 5))
plt.plot(y_test_inv, label='Actual')
plt.plot(preds_lstm, label='LSTM')
plt.plot(preds_gru, label='GRU')
plt.plot(preds_xgb, label='XGBoost')
plt.xlabel('Time Step')
plt.ylabel('Footfall')
plt.title('Model Predictions vs Actual')
plt.legend()
plt.tight_layout()
plt.show()


# %%
# !pip install xgboost



# %%
# Simple ensemble: average the three model predictions
preds_ensemble = (preds_lstm + preds_gru + preds_xgb) / 3

# Compute RMSE on the inverse‑scaled test set
from sklearn.metrics import mean_squared_error
import numpy as np

rmse_ensemble = np.sqrt(mean_squared_error(y_test_inv, preds_ensemble))
print(f"Ensemble RMSE: {rmse_ensemble:.2f}")


# %%
# Compute ensemble predictions (if not already done)
preds_ensemble = (preds_lstm + preds_gru + preds_xgb) / 3

# Plot Actual vs. Ensemble predictions
plt.figure(figsize=(12, 5))
plt.plot(y_test_inv, label='Actual')
plt.plot(preds_ensemble, label='Ensemble')
plt.xlabel('Time Step')
plt.ylabel('Footfall')
plt.title('Actual vs. Ensemble Predictions')
plt.legend()
plt.tight_layout()
plt.show()


# %%
from sklearn.metrics import mean_absolute_percentage_error

# Calculate Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(y_test_inv, preds_ensemble)

# Convert MAPE to “accuracy” (%) 
accuracy = 100 * (1 - mape)

print(f"Ensemble Accuracy: {accuracy:.2f}%")


# %%
import pandas as pd
import numpy as np

# Assuming lstm, gru, xgb, df, and scaler are defined outside this function
def predict_ensemble_for_datetime(dt_str, df, lstm, gru, xgb, scaler, n_steps=24):

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

# Example usage
save_models(lstm, gru, xgb)
lstm, gru, xgb = load_models()

print(predict_ensemble_for_datetime("29-03-2025 06:00", df, lstm, gru, xgb, scaler))

# %%
# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime

# # Set page configuration for a wide layout and custom title
# st.set_page_config(page_title="Footfall Analysis Dashboard", layout="wide")

# st.title("Footfall Analysis Dashboard")

# # --------------------------
# # Section 1: Overall Footfall Over Time
# # --------------------------
# st.header("1. Overall Footfall Over Time")
# fig1, ax1 = plt.subplots(figsize=(12, 5))
# # If df has datetime as its index (from your load_data function)
# ax1.plot(df.index, df['footfall'], color="tab:blue")
# ax1.set_xlabel("Date & Time")
# ax1.set_ylabel("Footfall")
# ax1.set_title("Footfall Over Time")
# plt.xticks(rotation=45)
# st.pyplot(fig1)

# # --------------------------
# # Section 2: Average Hourly Footfall by Day of Week
# # --------------------------
# st.header("2. Average Hourly Footfall by Day of Week (5:00–23:00)")
# # Ensure the day_of_week and hour columns exist; if not, create them
# if 'day_of_week' not in df.columns or 'hour' not in df.columns:
#     df['day_of_week'] = df.index.dayofweek
#     df['hour'] = df.index.hour

# df_filtered = df[(df['hour'] >= 5) & (df['hour'] <= 23)]
# avg = df_filtered.groupby(['hour', 'day_of_week'])['footfall'].mean().unstack()

# fig2, ax2 = plt.subplots(figsize=(12, 6))
# weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
# for dow in range(7):
#     if dow in avg.columns:
#         ax2.plot(avg.index, avg[dow], marker='o', label=weekday_names[dow])
# ax2.set_xticks(range(5, 24))
# ax2.set_xlabel("Hour of Day")
# ax2.set_ylabel("Average Footfall")
# ax2.set_title("Average Hourly Footfall by Day of Week (5:00–23:00)")
# ax2.legend(title="Day of Week")
# ax2.grid(True)
# st.pyplot(fig2)

# # --------------------------
# # Section 3: Ensemble Predictions vs Actual
# # --------------------------
# st.header("3. Ensemble Model Predictions vs Actual")
# fig3, ax3 = plt.subplots(figsize=(12, 5))
# ax3.plot(y_test_inv, label='Actual', color="tab:green")
# ax3.plot(preds_ensemble, label='Ensemble', color="tab:red")
# ax3.set_xlabel("Time Step")
# ax3.set_ylabel("Footfall")
# ax3.set_title("Actual vs. Ensemble Predictions")
# ax3.legend()
# st.pyplot(fig3)

# # --------------------------
# # Section 4: Predict Footfall for a Specific Date & Time
# # --------------------------
# st.header("4. Predict Footfall for a Specific Date & Time")
# selected_date = st.date_input("Select Date", value=datetime.now().date())
# selected_time = st.time_input("Select Time", value=datetime.now().time())

# if st.button("Predict Footfall"):
#     dt = datetime.combine(selected_date, selected_time)
#     dt_str = dt.strftime("%d-%m-%Y %H:%M")
#     prediction = predict_ensemble_for_datetime(dt_str)
#     st.success(f"Predicted Footfall for {dt_str} is: {prediction:.2f}")


# # %%
# # !streamlit run app.py &


# # %%
