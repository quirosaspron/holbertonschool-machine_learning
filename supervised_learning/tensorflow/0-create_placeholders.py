#!/usr/bin/env python3
"""Creates a place holder"""
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """ returns two placeholders """
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    return x, y
