#!/usr/bin/env python3
""" Creates a sparse autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Returns the encoder, decoder and autoencoder
    """
    # Encoder model
    input_layer = keras.layers.Input(shape=(input_dims,))
    encoded = input_layer
    for i in hidden_layers:
        encoded = keras.layers.Dense(i, activation='relu')(encoded)
    regularizer = keras.regularizers.l1(lambtha)
    latent = keras.layers.Dense(latent_dims, activation='relu',
                                activity_regularizer=regularizer)(encoded)
    encoder = keras.models.Model(inputs=input_layer, outputs=latent)

    # Decoder model
    latent_inputs = keras.layers.Input(shape=(latent_dims,))
    decoded = latent_inputs
    for i in reversed(hidden_layers):
        decoded = keras.layers.Dense(i, activation='relu')(decoded)
    output_layer = keras.layers.Dense(input_dims,
                                      activation='sigmoid')(decoded)
    decoder = keras.models.Model(inputs=latent_inputs,
                                 outputs=output_layer)

    # Autoencoder model
    autoencoder_input = input_layer
    autoencoder_output = decoder(encoder(autoencoder_input))
    autoencoder = keras.models.Model(inputs=autoencoder_input,
                                     outputs=autoencoder_output)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
