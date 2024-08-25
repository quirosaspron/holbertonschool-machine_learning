#!/usr/bin/env python3
""" Creates a vanilla autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Returns the encoder, decoder, and autoencoder
    """

    # Encoder
    input_layer = keras.layers.Input(shape=input_dims)

    encoder = input_layer
    for i in filters:
        encoder = keras.layers.Conv2D(i, (3, 3),
                                      padding='same',
                                      activation='relu')(encoder)
        encoder = keras.layers.MaxPool2D((2, 2),
                                         padding='same')(encoder)

    encoder = keras.Model(inputs=input_layer, outputs=encoder)

    # Decoder
    input_layer = keras.layers.Input(shape=latent_dims)

    decoder = input_layer
    for i in reversed(filters[1:]):
        decoder = keras.layers.Conv2D(i, (3, 3),
                                      padding='same',
                                      activation='relu')(decoder)
        decoder = keras.layers.UpSampling2D(size=(2, 2))(decoder)

    decoder = keras.layers.Conv2D(filters[0], (3, 3),
                                  padding='valid',
                                  activation='relu')(decoder)
    decoder = keras.layers.UpSampling2D(size=(2, 2))(decoder)

    decoder = keras.layers.Conv2D(input_dims[-1], (3, 3),
                                  padding='same',
                                  activation='sigmoid')(decoder)

    decoder = keras.models.Model(inputs=input_layer, outputs=decoder)

    # Autoencoder

    encoder_input = keras.layers.Input(shape=input_dims)
    encoded_output = encoder(encoder_input)
    decoded_output = decoder(encoded_output)

    autodecoder = keras.Model(inputs=encoder_input, outputs=decoded_output)
    autodecoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autodecoder
