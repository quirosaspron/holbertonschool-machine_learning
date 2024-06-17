#!/usr/bin/env python3
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    he_normal = K.initializers.HeNormal(seed=0)

    for i in range(layers):
        # Batch Normalization and ReLU activation
        bn_1 = K.layers.BatchNormalization(axis=-1)(X)
        relu_1 = K.layers.Activation('relu')(bn_1)
        # 1x1 Convolution (Bottleneck layer)
        conv_1 = K.layers.Conv2D(4 * growth_rate, (1, 1),
                                 padding='same',
                                 kernel_initializer=he_normal)(relu_1)
        # Batch Normalization and ReLU activation
        bn_2 = K.layers.BatchNormalization(axis=-1)(conv_1)
        relu_2 = K.layers.Activation('relu')(bn_2)
        # 3x3 Convolution
        conv_2 = K.layers.Conv2D(growth_rate, (3, 3), padding='same',
                                 kernel_initializer=he_normal)(relu_2)
        # Concatenate the input with the output of the convolution
        X = K.layers.Concatenate(axis=-1)([X, conv_2])
        # Update the number of filters
        nb_filters += growth_rate
    return X, nb_filters
