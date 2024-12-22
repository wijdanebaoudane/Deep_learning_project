from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
import yfinance as yf
import os

# Define the model file path
MODEL_PATH = "model_LSTM_baoudane.h5"

# Verify if the model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Load the trained model
model = load_model(MODEL_PATH)

# Initialize FastAPI app
app = FastAPI()

# Define Pydantic models for input and output
class StockRequest(BaseModel):
    symbol: str
    start_date: str
    end_date: str

class PredictionResponse(BaseModel):
    actual_prices: list
    predicted_prices: list
    dates: list

# Define helper function for processing and predicting stock prices
@app.post("/predict", response_model=PredictionResponse)
async def predict_stock_price(request: StockRequest):
    try:
        # Fetch stock data using yfinance
        data = yf.download(request.symbol, start=request.start_date, end=request.end_date)
        
        if 'Close' not in data.columns:
            raise HTTPException(status_code=400, detail="Data does not contain 'Close' column.")

        # Preprocess the data (scaling and shaping)
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data[['Close']])

        # Create sequences for prediction
        time_step = 60
        X = []
        for i in range(time_step, len(data_scaled)):
            X.append(data_scaled[i - time_step:i, 0])
        X = np.array(X).reshape(-1, time_step, 1)

        # Make predictions
        predictions = model.predict(X)
        predictions = scaler.inverse_transform(predictions)

        # Prepare the response
        predicted_prices = predictions.flatten().tolist()
        actual_prices = data['Close'].iloc[time_step:].tolist()
        dates = data.index[time_step:].strftime('%Y-%m-%d').tolist()

        return PredictionResponse(actual_prices=actual_prices, predicted_prices=predicted_prices, dates=dates)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Stock Price Prediction API!"}
