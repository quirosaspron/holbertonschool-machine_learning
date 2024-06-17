#!/usr/bin/env python3
"""builds an inception block"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    A_prev is the output from the previous layer
    filters is a tuple or list containing
    F1, F3R, F3,F5R, F5, FPP, respectively:
    F1 is the number of filters in the 1x1 convolution
    F3R is the number of filters in the 1x1 convolution
    before the 3x3 convolution
    F3 is the number of filters in the 3x3 convolution
    F5R is the number of filters in the 1x1 convolution
    before the 5x5 convolution
    F5 is the number of filters in the 5x5 convolution
    FPP is the number of filters in the 1x1 convolution
    after the max pooling
    All convolutions inside the inception block should
    use a rectified linear activation (ReLU)
    Returns: the concatenated output of the inception block
    """
    # Defining branch1
    x1 = K.layers.Conv2D(filters[0], (1, 1),
                         padding='same',
                         activation='relu')(A_prev)
    # Defining branch2
    prev_x2 = K.layers.Conv2D(filters[1], (1, 1),
                              padding='same',
                              activation='relu')(A_prev)
    x2 = K.layers.Conv2D(filters[2], (3, 3),
                         padding='same',
                         activation='relu')(prev_x2)
    # Defining branch3
    prev_x3 = K.layers.Conv2D(filters[3], (1, 1),
                              padding='same',
                              activation='relu')(A_prev)
    x3 = K.layers.Conv2D(filters[4], (5, 5),
                         padding='same',
                         activation='relu')(prev_x3)
    # Defining branch4
    prev_x4 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                    strides=(1, 1),
                                    padding='same')(A_prev)
    x4 = K.layers.Conv2D(filters[5], (1, 1),
                         padding='same',
                         activation='relu')(prev_x4)
    # Concatenate all the branches
    return K.layers.Concatenate(axis=-1)([x1, x2, x3, x4])
