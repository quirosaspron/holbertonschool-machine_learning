#!/usr/bin/env python3
"""Builds an inception network"""
from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    A_prev is the output from the previous layer
    filters is a tuple or list containing
    F11, F3, F12, respectively:
    F11 is the number of filters in the first 1x1 convolution
    F3 is the number of filters in the 3x3 convolution
    F12 is the number of filters in the second 1x1 convolution
    All convolutions inside the block should be followed by batch
    normalization along the channels axis
    and a rectified linear activation (ReLU), respectively.
    All weights should use he normal initialization
    The seed for the he_normal initializer should be set to zero
    Returns: the activated output of the identity block
    """
    input_tensor = K.layers.Input(shape=(224, 224, 3))
    conv_1 = K.layers.Conv2D(64, (7, 7), padding='same',
                             activation='relu', strides=(2, 2))(input_tensor)
    max_pool_1 = K.layers.MaxPooling2D((3, 3), padding='same',
                                       strides=(2, 2))(conv_1)
    conv_2 = K.layers.Conv2D(64, (1, 1), activation='relu')(max_pool_1)
    conv_3 = K.layers.Conv2D(192, (3, 3), activation='relu')(conv_2)
    max_pool_2 = K.layers.MaxPooling2D((3, 3), padding='same',
                                       strides=(2, 2))(conv_3)
    inception_1 = inception_block(max_pool_2,
                                  [64, 96, 128, 16, 32, 32])
    inception_2 = inception_block(inception_1,
                                  [128, 128, 192, 32, 96, 64])
    max_pool_3 = K.layers.MaxPooling2D((3, 3), padding='same',
                                       strides=(2, 2))(inception_2)
    inception_3 = inception_block(max_pool_3,
                                  [192, 96, 208, 16, 48, 64])
    inception_4 = inception_block(inception_3,
                                  [160, 112, 224, 24, 64, 64])
    inception_5 = inception_block(inception_4,
                                  [128, 128, 256, 24, 64, 64])
    inception_6 = inception_block(inception_5,
                                  [112, 144, 288, 32, 64, 64])
    inception_7 = inception_block(inception_6,
                                  [256, 160, 320, 32, 128, 128])
    max_pool_4 = K.layers.MaxPooling2D((3, 3), padding='same',
                                       strides=(2, 2))(inception_7)
    inception_8 = inception_block(max_pool_4,
                                  [256, 160, 320, 32, 128, 128])
    inception_9 = inception_block(inception_8,
                                  [384, 192, 384, 48, 128, 128])
    avg_pool = K.layers.AveragePooling2D((7, 7))(inception_9)
    dropout = K.layers.Dropout(0.4)(avg_pool)
    output_tensor = K.layers.Dense(1000, activation='softmax')(dropout)
    model = K.models.Model(inputs=input_tensor, outputs=output_tensor)
    return model
