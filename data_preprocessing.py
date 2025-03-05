import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# Fetching historical stock data
ticker = 'GOOGL'
data = yf.download(ticker, start='2015-01-01', end='2025-01-01')
data = data[['Close']]  # We only need the closing prices

# Check if data was fetched successfully
if data.empty:
    raise ValueError("No data fetched for the given ticker and date range.")
else:
    print(f"Data fetched successfully. Total records: {len(data)}")

# Plot the closing prices
plt.figure(figsize=(10,5))
plt.plot(data, label='Close Price History')
plt.xlabel('Date')
plt.ylabel('Close Price USD')
plt.title(f'{ticker} Stock Price History')
plt.legend()
plt.show()

# Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

sequence_length = 60  # Using the last 60 days to predict the next day's price
X, y = [], []

# Sequence Generation Check
if len(scaled_data) < sequence_length:
    raise ValueError("Not enough data points for the given sequence length.")
else:
    print(f"Total data points after scaling: {len(scaled_data)}")

# Create sequences
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Check shapes before saving
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Save the preprocessed data
np.save('X.npy', X)
np.save('y.npy', y)

# Confirm files are saved and not empty
if os.path.exists('X.npy') and os.path.exists('y.npy'):
    X_loaded = np.load('X.npy')
    y_loaded = np.load('y.npy')
    print(f"Loaded X shape: {X_loaded.shape}, y shape: {y_loaded.shape}")
    if X_loaded.size == 0 or y_loaded.size == 0:
        raise ValueError("Saved files are empty.")
    else:
        print("Data saved and verified successfully.")
else:
    raise FileNotFoundError("X.npy or y.npy not found.")
