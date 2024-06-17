#!/usr/bin/env python3
"""Builds a projection block"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in
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
                Number of filters in the second 1x1 convolution as well
                as the 1x1 convolution in the shortcut connection.
    s : int
        Stride of the first convolution in both the main path and the shortcut
        connection.

    Returns:
    tensor
        The activated output of the projection block.
    """
    F11, F3, F12 = filters
    init = K.initializers.HeNormal(seed=0)
    layer_1 = K.layers.Conv2D(F11, (1, 1), strides=(s, s),
                              padding='same',
                              kernel_initializer=init)(A_prev)
    layer_1 = K.layers.BatchNormalization(axis=-1)(layer_1)
    layer_1 = K.layers.Activation('relu')(layer_1)

    layer_2 = K.layers.Conv2D(F3, (3, 3), padding='same',
                              kernel_initializer=init)(layer_1)
    layer_2 = K.layers.BatchNormalization(axis=-1)(layer_2)
    layer_2 = K.layers.Activation('relu')(layer_2)

    layer_3 = K.layers.Conv2D(F12, (1, 1), padding='same',
                              kernel_initializer=init)(layer_2)
    layer_3 = K.layers.BatchNormalization(axis=-1)(layer_3)
    shortcut_layer = K.layers.Conv2D(F12, (1, 1), strides=(s, s),
                                     padding='same',
                                     kernel_initializer=init)(A_prev)
    shortcut_layer = K.layers.BatchNormalization(axis=-1)(shortcut_layer)
    # Merge output of main path and shortcut path
    merged = K.layers.Add()([layer_3, shortcut_layer])

    # Return activated output of merge, using ReLU.
    return K.layers.Activation(activation="relu")(merged)
