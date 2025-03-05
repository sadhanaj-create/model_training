
# A Machine Learning Project with Docker Deployment  

**Use Case**: Stock Price Prediction using LSTM  

This project aims to predict future stock prices using a Long Short-Term Memory (**LSTM**) model. The model takes historical stock price data as input and forecasts the next day's closing price. The implementation includes data preprocessing, model training, evaluation, and deployment as a Flask API inside a Docker container. It also includes **unit and integration tests** for the entire ML pipeline.

---

### **Project Features**:
1. **Data Preprocessing**: Fetch historical stock data from Yahoo Finance (via `yfinance`), normalize stock prices to improve model performance, and create time-series sequences (saved as `X.npy` and `y.npy`) for LSTM input.
2. **Model Building**: Design and build an **LSTM** model using the **PyTorch** framework.
3. **Model Training**: Train the model on the preprocessed stock price data.
4. **Real-time Prediction**: Deploy the trained model using a **Flask API**, which predicts the next day's stock price and returns the prediction as a JSON response.
5. **Deployment**: The project is deployed in a **Docker** container, making it easy to run in any environment.
6. **Testing**: Includes unit and integration tests to ensure the quality and reliability of the system.

---

### **Project Structure**:
```
/stock-predictor
│── app.py                    # Flask API for stock prediction
│── data_preprocessing.py     # Data fetching and preprocessing
│── model.py                  # LSTM Model definition and training
│── Dockerfile                # Docker setup
│── requirements.txt          # Dependencies
│── tests/                    # Test directory
│   ├── test_api.py           # API endpoint tests
│   ├── test_data.py          # Data preprocessing tests
│   ├── test_model.py         # Model training & inference tests
│── X.npy                     # Processed input data
│── y.npy                     # Processed target data
│── README.md                 # Project documentation
```

---

### **Tech Stack**:
- **Python**
- **PyTorch** (Deep Learning Framework)
- **Pandas** & **NumPy** (Data Processing)
- **Scikit-Learn** (Feature Engineering)
- **Matplotlib** (Visualization)
- **Flask** (API Deployment)
- **Docker** (Containerization)
- **Pytest** (Testing)

---

### **Docker Deployment**:

1️⃣ **Build Docker Image**  
```bash
docker build -t stock-predictor:latest .
```

2️⃣ **Run the Container**  
```bash
docker run -p 8080:8080 stock-predictor
```

3️⃣ **Test the API**  
```bash
curl -X POST -H "Content-Type: application/json"     -d '{"ticker": "GOOGL"}'     http://localhost:8080/predict
```

---

### **🧪 Testing**:

1️⃣ **Running Tests Locally**  
```bash
pip install -r requirements.txt
pytest tests/
```

2️⃣ **Running Tests in Docker**  
```bash
docker run --rm stock-predictor pytest tests/
```

---

### **License**:
This project is licensed under the **MIT License**.
