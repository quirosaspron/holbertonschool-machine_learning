#!/usr/bin/env python3
""" Creates a vanilla autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Returns the encoder, decoder and autoencoder
    """
    encoder = keras.models.Sequential()
    encoder.add(keras.layers.Input(shape=(input_dims,)))
    for i in hidden_layers:
        encoder.add(keras.layers.Dense(i, activation='relu'))
    encoder.add(keras.layers.Dense(latent_dims, activation='relu'))

    decoder = keras.models.Sequential()
    decoder.add(keras.layers.Input(shape=(latent_dims, )))
    for i in reversed(hidden_layers):
        decoder.add(keras.layers.Dense(i, activation='relu'))
    decoder.add(keras.layers.Dense(input_dims, activation='sigmoid'))

    autoencoder = keras.models.Sequential()
    autoencoder.add(encoder)
    autoencoder.add(decoder)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
