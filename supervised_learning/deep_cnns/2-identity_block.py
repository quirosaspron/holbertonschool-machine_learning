#!/usr/bin/env python3
"""Builds an indentity block"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in
    'Deep Residual Learning for Image Recognition' (2015).

    Parameters:
    A_prev : tensor
        The output of the previous layer.
    filters : tuple or list
        Contains F11, F3, F12 respectively:
            F11 : int
                Number of filters in the first 1x1 convolution.
            F3 : int
                Number of filters in the 3x3 convolution.
            F12 : int
            Number of filters in the second 1x1 convolution.

    Returns:
    tensor
        The activated output of the identity block.
    """
    init = K.initializers.HeNormal(seed=0)
    layer_1 = K.layers.Conv2D(filters[0], (1, 1), padding='same',
                              kernel_initializer=init)(A_prev)
    layer_1 = K.layers.BatchNormalization(axis=-1)(layer_1)
    layer_1 = K.layers.Activation('relu')(layer_1)

    layer_2 = K.layers.Conv2D(filters[1], (3, 3), padding='same',
                              kernel_initializer=init)(layer_1)
    layer_2 = K.layers.BatchNormalization(axis=-1)(layer_2)
    layer_2 = K.layers.Activation('relu')(layer_2)

    layer_3 = K.layers.Conv2D(filters[2], (1, 1), padding='same',
                              kernel_initializer=init)(layer_2)
    layer_3 = K.layers.BatchNormalization(axis=-1)(layer_3)
    layer_3 = K.layers.Activation('relu')(layer_3 + A_prev)

    return layer_3
