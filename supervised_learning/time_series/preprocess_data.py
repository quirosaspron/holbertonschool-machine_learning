import pandas as pd
import numpy as np
import tensorflow as tf
from data_visualization import plot_closing_price

def preprocess_data(coinbase_file, bitstamp_file):
    # Load the Coinbase and Bitstamp datasets
    coinbase = pd.read_csv(coinbase_file)
    bitstamp = pd.read_csv(bitstamp_file)

    # Change time from string to date type and set it as index
    coinbase = coinbase.set_index(pd.to_datetime(coinbase['Timestamp'], unit='s'))
    bitstamp = bitstamp.set_index(pd.to_datetime(bitstamp['Timestamp'], unit='s'))

    coinbase = coinbase.drop('Timestamp', axis=1)
    bitstamp = bitstamp.drop('Timestamp', axis=1)

    # Combine the datasets and fill the gaps
    data = coinbase.combine_first(bitstamp)
    data.ffill(inplace=True)


    plot_closing_price(data)

    # Drop data before 2017
    data = data[data.index >= '2017-01-01']

    plot_closing_price(data)

    # Resample the data with hours
    data = data.resample('h').mean()

    # Drop the weighted price column
    data.drop(columns=["Weighted_Price"])

    # Make the data stationary with differential logarithm
    data['Close'] = np.log(data['Close'])  # Log transformation
    data['Close'] = data['Close'].diff() # Differencing
    data = data.dropna()

    # Clip values to be within [-0.10, 0.10]
    data['Close'] = data['Close'].clip(lower=-0.10, upper=0.10)

    plot_closing_price(data)
    # Slice data into windows
    sequences = []
    labels = []
    for i in range(len(data) - 24):
        sequences.append(data.iloc[i:i + 24].values)  # Use .values for NumPy array
        labels.append(data.iloc[i + 24][3])  # Use the 'close' price as the label

    sequences = np.array(sequences)
    labels = np.array(labels)

    return sequences, labels

