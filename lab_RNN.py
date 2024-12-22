import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split

# Step 1: Fetch the stock data from Yahoo Finance
def fetch_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

# Step 2: Prepare the data
def prepare_data(df):
    # Use 'Close' price for prediction
    data = df['Close'].values
    data = data.reshape(-1, 1)

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Prepare training data (use previous 60 days to predict the next day)
    x_data, y_data = [], []
    for i in range(60, len(scaled_data)):
        x_data.append(scaled_data[i-60:i, 0])
        y_data.append(scaled_data[i, 0])

    x_data, y_data = np.array(x_data), np.array(y_data)

    # Reshape data for RNN input: [samples, time steps, features]
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))

    return x_data, y_data, scaler

# Step 3: Build the RNN model
def build_rnn_model(input_shape):
    model = Sequential()

    # Add LSTM layer with 50 units and Dropout regularization
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    # Add another LSTM layer
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    # Add Dense layer for output
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Step 4: Train the model
def train_rnn_model(model, x_data, y_data, epochs=10, batch_size=32):
    model.fit(x_data, y_data, epochs=epochs, batch_size=batch_size)

# Step 5: Predict stock prices and visualize the results
def predict_and_plot(df, model, scaler, ticker):
    # Get the last 60 days of data for prediction
    test_data = df['Close'].values
    test_data = test_data[-60:].reshape(-1, 1)
    scaled_test_data = scaler.transform(test_data)

    x_test = []
    x_test.append(scaled_test_data)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Predict the stock price
    predicted_price = model.predict(x_test)
    predicted_price = scaler.inverse_transform(predicted_price)

    # Plot the real vs predicted stock price
    plt.figure(figsize=(14, 8))
    plt.plot(df['Close'], label=f'Real {ticker} Stock Price')
    plt.plot(np.arange(len(df), len(df) + 1), predicted_price, label=f'Predicted {ticker} Stock Price', linestyle='--')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Example Usage
if __name__ == "__main__":
    ticker = 'AAPL'  # Apple stock symbol, change as needed
    start_date = '2015-01-01'
    end_date = '2024-01-01'

    # Fetch and prepare data
    df = fetch_stock_data(ticker, start_date, end_date)
    x_data, y_data, scaler = prepare_data(df)

    # Build and train the model
    model = build_rnn_model((x_data.shape[1], 1))
    train_rnn_model(model, x_data, y_data, epochs=10, batch_size=32)

    # Predict and plot results
    predict_and_plot(df, model, scaler, ticker)
