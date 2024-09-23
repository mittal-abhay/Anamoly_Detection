import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import yfinance as yf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense

def synthetic_stock_price_stream(mean=100, seasonal_variation=5, noise_level=1.0, trend=0.1, volatility=1.0):
    t = 0
    price = mean
    while True:
        seasonal_effect = seasonal_variation * np.sin(2 * np.pi * t / 50)
        trend_effect = trend * t
        noise = np.random.normal(0, noise_level)
        jump = np.random.normal(0, volatility) if np.random.rand() < 0.1 else 0
        price += noise + seasonal_effect + trend_effect + jump
        yield price
        t += 1

def fetch_real_stock_data(ticker='VOW3', period='1y', interval='1d'):
    stock_data = yf.download(ticker, period=period, interval=interval)
    return stock_data


def collect_normal_data(data_stream, num_samples=1000, time_steps=10):
    X_train = []
    window = []
    
    for i, data_point in enumerate(data_stream):
        if data_point is not None:
            window.append([data_point])
            if len(window) == time_steps:
                X_train.append(window)
                window = []
            if len(X_train) == num_samples:
                break
    return np.array(X_train)

def real_time_plot_with_dynamic_threshold(data_stream, stock_dates, model, time_steps=10, smoothing_window=60):
    window = deque(maxlen=time_steps)
    error_window = deque(maxlen=smoothing_window)

    x_data, y_data = [], []
    anomaly_x, anomaly_y = [], []
    anomaly_ranges = []
    prediction_y = []
    smoothed_prediction_y = []

    anomaly_start = None

    for idx, data_point in enumerate(data_stream):
        if data_point is not None:
            window.append([data_point])

            if len(window) == time_steps:
                window_array = np.array(window).reshape(1, time_steps, 1)
                reconstructed = model.predict(window_array)
                error = np.mean(np.abs(window_array - reconstructed))
                error_window.append(error)

                x_data.append(stock_dates[idx])  # Use real date for x-axis
                y_data.append(data_point)
                prediction_y.append(reconstructed[0, -1, 0])

                # Detect anomaly and update anomaly ranges and points
                if error > np.mean(error_window) + 3 * np.std(error_window):
                    anomaly_x.append(stock_dates[idx])  # Use real date for x-axis
                    anomaly_y.append(data_point)
                    if anomaly_start is None:
                        anomaly_start = idx
                else:
                    if anomaly_start is not None:
                        anomaly_ranges.append((anomaly_start, idx - 1))
                        anomaly_start = None

                # Calculate smoothed prediction
                if len(prediction_y) >= smoothing_window:
                    smoothed_value = np.mean(prediction_y[-smoothing_window:])
                else:
                    smoothed_value = np.mean(prediction_y)
                smoothed_prediction_y.append(smoothed_value)

    # Plotting the final static figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_data, y_data, color='blue', label='Actual Stock Prices')
    ax.plot(x_data, smoothed_prediction_y, color='green', label='Smoothed Predicted Prices', alpha=0.7)

    # Plot red ranges for anomalies
    for start, end in anomaly_ranges:
        ax.axvspan(stock_dates[start], stock_dates[end], color='red', alpha=0.3)
    
    # If there's an ongoing anomaly, plot it up to the last point
    if anomaly_start is not None:
        ax.axvspan(stock_dates[anomaly_start], stock_dates[-1], color='red', alpha=0.3)

    # Plot red points for individual anomalies
    ax.scatter(anomaly_x, anomaly_y, color='red', marker='o', label='Anomalies')

    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')
    ax.set_title('Final Stock Prices with Anomaly Ranges and Points')
    ax.legend()
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))  # Show a maximum of 10 date labels
    plt.xticks(rotation=45)  # Rotate date labels for better readability
    
    # Save final static plot
    plt.savefig('images/final_stock_anomaly_detection.png')
    plt.show()


def plot_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 100)
    plt.savefig('images/loss_plot.png')
    plt.show()

