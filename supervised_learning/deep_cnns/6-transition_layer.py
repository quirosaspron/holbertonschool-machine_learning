#!/usr/bin/env python3
"""Builds a transition layer"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in
    'Densely Connected Convolutional Networks' (2016).

    Parameters:
    X : tensor
        The output of the previous layer.
    nb_filters : int
        The number of filters in X.
    compression : float
        The compression factor for the transition layer.

    Returns:
    tensor, int
        The output of the transition layer and the number of filters
        within the output, respectively.
    """
    he_normal = K.initializers.HeNormal(seed=0)
    # Apply Batch Normalization
    X = K.layers.BatchNormalization(axis=-1)(X)
    # Apply ReLU activation
    X = K.layers.Activation('relu')(X)
    # Compute the number of filters after compression
    compressed_filters = int(nb_filters * compression)
    # Apply 1x1 Convolution
    X = K.layers.Conv2D(compressed_filters, (1, 1),
                        padding='same', kernel_initializer=he_normal)(X)
    # Apply Average Pooling
    X = K.layers.AveragePooling2D(pool_size=(2, 2),
                                  strides=(2, 2), padding='same')(X)
    return X, compressed_filters
