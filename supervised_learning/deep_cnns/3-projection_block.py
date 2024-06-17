#!/usr/bin/env python3
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    init = K.initializers.HeNormal(seed=0)
    layer_1 = K.layers.Conv2D(filters[0], (1, 1), strides=(s, s),
                              padding='same',
                              kernel_initializer=init)(A_prev)
    layer_1 = K.layers.BatchNormalization()(layer_1)
    layer_1 = K.layers.Activation('relu')(layer_1)

    layer_2 = K.layers.Conv2D(filters[1], (3, 3), padding='same',
                              kernel_initializer=init)(layer_1)
    layer_2 = K.layers.BatchNormalization()(layer_2)
    layer_2 = K.layers.Activation('relu')(layer_2)

    layer_3 = K.layers.Conv2D(filters[2], (1, 1), padding='same',
                              kernel_initializer=init)(layer_2)
    layer_3 = K.layers.BatchNormalization()(layer_3)
    shortcut_layer = K.layers.Conv2D(filters[2], (1, 1), strides=(s, s),
                                     padding='same',
                                     kernel_initializer=init)(A_prev)
    shortcut_layer = K.layers.BatchNormalization()(shortcut_layer)
    layer_3 = K.layers.Activation('relu')(layer_3 + shortcut_layer)

    return layer_3
