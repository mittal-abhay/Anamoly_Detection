from src.utils import *
from src.model import build_lstm_autoencoder
from tensorflow.keras.models import load_model
import os


# Fetch real stock prices and dates
stock_data = fetch_real_stock_data('AAPL', period='10y', interval='1d')
real_stock_prices = stock_data['Close'].values
stock_dates = stock_data.index  # Get dates from the stock data

stream = (price for price in real_stock_prices)

input_shape = (10, 1)
X_train = collect_normal_data(stream, num_samples=3000, time_steps=10)


#if model exists load it, else create it
if os.path.exists("models/lstm_autoencoder_model.keras"):
    lstm_autoencoder = load_model("models/lstm_autoencoder_model.keras")
else:
    # Build and train the LSTM Autoencoder Model
    lstm_autoencoder = build_lstm_autoencoder(input_shape)
    split = int(0.8 * len(X_train))
    X_train_data = X_train[:split]
    X_val_data = X_train[split:]

    history = lstm_autoencoder.fit(
        X_train_data, X_train_data,
        epochs=500,
        batch_size=64,
        validation_data=(X_val_data, X_val_data),
        shuffle=True
    )

    # Save and load the model
    lstm_autoencoder.save("models/lstm_autoencoder_model.keras")
    lstm_autoencoder = load_model("models/lstm_autoencoder_model.keras")

    plot_loss(history)

# Reset the stream for real-time detection
real_stock_prices = stock_data['Close'].values
stream = (price for price in real_stock_prices)

# Run real-time analysis with dynamic threshold and dates
real_time_plot_with_dynamic_threshold(stream, stock_dates, lstm_autoencoder, time_steps=10, smoothing_window=60)
