import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error

# Function to download stock data
def download_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    return data[['Close']]

# Function to create sequences for LSTM input
def create_sequences(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i - time_step:i, 0])  # Use previous 'time_step' days
        y.append(data[i, 0])  # Predict the next day's closing price
    return np.array(X), np.array(y)

# Main function
def main():
    stock_symbol = 'AAPL'
    start_date = '2015-01-01'
    end_date = '2024-11-21'
    time_step = 60  # Number of previous days to consider
    batch_size = 32  # Batch size for training
    epochs = 15  # Number of epochs

    # Step 1: Download stock data
    data_close = download_data(stock_symbol, start_date, end_date)

    # Step 2: Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_close)

    # Step 3: Create sequences for training/testing
    X, y = create_sequences(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for LSTM input

    # Step 4: Split the data into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Step 5: Build the LSTM model
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.3),
        LSTM(units=50, return_sequences=False),
        Dropout(0.3),
        Dense(units=25),
        Dense(units=1)
    ])

    # Step 6: Compile and train the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    # Save the trained model
    model.save('model_LSTM_baoudane.h5')

    # Step 7: Make predictions
    predicted_stock_price = model.predict(X_test)

    # Step 8: Inverse transform the predictions and actual values
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Step 9: Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test_actual, predicted_stock_price))
    print(f'Root Mean Squared Error: {rmse}')

    # Step 10: Save predictions to a CSV file
    predictions_df = pd.DataFrame({
        'Actual': y_test_actual.flatten(),
        'Predicted': predicted_stock_price.flatten()
    })
    predictions_df.to_csv('stock_predictions.csv', index=False)

    # Step 11: Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_actual, color='blue', label=f'Actual {stock_symbol} Stock Price')
    plt.plot(predicted_stock_price, color='orange', label=f'Predicted {stock_symbol} Stock Price')
    plt.title(f'{stock_symbol} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
