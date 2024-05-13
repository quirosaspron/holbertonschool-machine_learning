#!/usr/bin/env python3
"""creates the training operation for the network"""
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def create_train_op(loss, alpha):
    """ creates the training operation"""
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    train_op = optimizer.minimize(loss)
    return train_op
