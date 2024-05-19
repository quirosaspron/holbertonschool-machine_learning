#!/usr/bin/env python3
"""sets up gd with momentum in TensorFlow"""
import numpy as np
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """Returns: optimizer"""
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=alpha,
        momentum=beta1)

    return optimizer
