import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from preprocess_data import preprocess_data
import data_visualization
sequences, labels = preprocess_data('coinbase.csv', 'bitstamp.csv')
dataset = tf.data.Dataset.from_tensor_slices((sequences, labels))
model = Sequential()


# Split into training, validation, and test sets
train_size = int(len(sequences) * 0.7)  # 70% for training
val_size = int(len(sequences) * 0.15)    # 15% for validation

train_sequences = sequences[:train_size]
train_labels = labels[:train_size]
val_sequences = sequences[train_size:train_size + val_size]
val_labels = labels[train_size:train_size + val_size]
test_sequences = sequences[train_size + val_size:]
test_labels = labels[train_size + val_size:]

# Normalize each dataset independently
scaler = MinMaxScaler(feature_range=(0, 1))
train_sequences = scaler.fit_transform(train_sequences.reshape(-1, train_sequences.shape[-1])).reshape(train_sequences.shape)
val_sequences = scaler.transform(val_sequences.reshape(-1, val_sequences.shape[-1])).reshape(val_sequences.shape)
test_sequences = scaler.transform(test_sequences.reshape(-1, test_sequences.shape[-1])).reshape(test_sequences.shape)



# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_labels)).batch(32)
val_dataset = tf.data.Dataset.from_tensor_slices((val_sequences, val_labels)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((test_sequences, test_labels)).batch(32)


# Define input shape based on the sequences
sequence_length = sequences.shape[1]
num_features = sequences.shape[2] # Number of features

# Design model architecture
# Add LSTM layers
model.add(LSTM(100, return_sequences=True, input_shape=(sequence_length, num_features)))
model.add(LSTM(100))
# Output layer
model.add(Dense(1))  # Predicting the 'close' price
# Compile the model with MSE loss function
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(train_dataset, epochs=50, validation_data=val_dataset)
# Visualize the losses
data_visualization.plot_losses(history)
model.save('lstm_model.kearas')

# Evaluate the model
test_loss = model.evaluate(test_dataset)
print("Test Loss:", test_loss)

y_pred = model.predict(test_dataset)

data_visualization.plot_predictions(test_labels, y_pred)
