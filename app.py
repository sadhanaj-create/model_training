from flask import Flask, request, jsonify
import torch
import numpy as np
from model import LSTMStockPredictor
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import subprocess

# Before making the prediction
subprocess.run(['python3', 'data_preprocessing.py'])

app = Flask(__name__)
model = LSTMStockPredictor()
model.load_state_dict(torch.load('lstm_stock_model.pth'))
model.eval()

scaler = MinMaxScaler(feature_range=(0, 1))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    ticker = data['ticker']
    stock_data = yf.download(ticker, period='75d')
    close_prices = stock_data['Close'].values[-60:]
    scaled_input = scaler.fit_transform(close_prices.reshape(-1, 1))
    X = torch.tensor(scaled_input.reshape(1, 60, 1), dtype=torch.float32)
    with torch.no_grad():
        prediction = model(X).item()
    predicted_price = scaler.inverse_transform(np.array([[prediction]]))
    return jsonify({'predicted_price': predicted_price[0][0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
