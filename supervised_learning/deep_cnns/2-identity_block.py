#!/usr/bin/env python3
import tensorflow.keras as K


def identity_block(A_prev, filters):
    init = K.initializers.HeNormal(seed=0)
    layer_1 = K.layers.Conv2D(filters[0], (1, 1), padding='same',
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
    layer_3 = K.layers.Activation('relu')(layer_3 + A_prev)

    return layer_3
