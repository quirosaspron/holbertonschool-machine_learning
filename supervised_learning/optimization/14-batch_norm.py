#!/usr/bin/env python3
"""creates a batch normalization layer for a neural network
in tensorflow"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Returns: a tensor of the activated output for the layer"""
    model = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode="fan_avg"),
        use_bias=False,
        )(prev)
    batch_mean, batch_var = tf.nn.moments(model, [0])

    gamma = tf.Variable(tf.ones([n]))
    beta = tf.Variable(tf.zeros([n]))

    norm = tf.nn.batch_normalization(
        model, batch_mean, batch_var, beta, gamma, 1e-7)

    return activation(norm)
