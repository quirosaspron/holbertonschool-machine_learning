#!/usr/bin/env python3
"""Creates a layer"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def create_layer(prev, n, activation):
    """ Returns the tensor output of a layer """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init, name='layer')
    return layer(prev)
