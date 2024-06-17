#!/usr/bin/env python3

from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
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
