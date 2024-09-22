import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_closing_price(data):
    """
    Plot the historical closing price data.

    Parameters:
    - data: DataFrame, dataset containing closing price data.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Closing Price', color='blue')
    plt.title('Historical Closing Price Data')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid()
    plt.show()

def plot_losses(history):
    """
    Plot the training and validation losses during model training.

    Parameters:
    - history: History object, contains loss information from model training.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(history.history['loss'], label='Training Loss', color='red')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='green')
    plt.title('Model Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

def plot_predictions(y_true, y_pred):
    """
    Compare actual closing prices with predicted closing prices.

    Parameters:
    - y_true: array-like, true closing prices for the test set.
    - y_pred: array-like, predicted closing prices from the model.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(y_true, label='Actual Closing Prices', color='blue')
    plt.plot(y_pred, label='Predicted Closing Prices', color='orange')
    plt.title('Comparison of Actual vs Predicted Closing Prices')
    plt.xlabel('Time Steps')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid()
    plt.show()
