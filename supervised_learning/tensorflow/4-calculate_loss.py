#!/usr/bin/env python3
"""Calculates the loss of the predicitons"""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """softmax cross-entropy loss function"""
    cross_entropy_loss = tf.compat.v1.losses.softmax_cross_entropy(
           onehot_labels=y, logits=y_pred)
    return cross_entropy_loss
