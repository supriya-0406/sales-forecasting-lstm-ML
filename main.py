import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset
df = pd.read_csv("sales_data.csv")

# Prepare data
data = df['Sales'].values.reshape(-1,1)

# Normalize
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create sequences (window size = 3)
X = []
y = []

for i in range(3, len(data_scaled)):
    X.append(data_scaled[i-3:i])
    y.append(data_scaled[i])

X = np.array(X)
y = np.array(y)

# Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X.shape[1], 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train
model.fit(X, y, epochs=100, verbose=0)

# Predict
predicted = model.predict(X)

# Convert back
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(y)

# Plot
plt.plot(actual_prices, label="Actual Sales")
plt.plot(predicted_prices, label="Predicted Sales")
plt.legend()
plt.title("Sales Forecasting using LSTM")
plt.xlabel("Time")
plt.ylabel("Sales")
plt.show()

# Predict next value
last_sequence = data_scaled[-3:]
last_sequence = last_sequence.reshape(1,3,1)

next_pred = model.predict(last_sequence)
next_value = scaler.inverse_transform(next_pred)

print("\nNext Day Sales Prediction:", next_value[0][0])
