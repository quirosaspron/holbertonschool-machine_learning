#!/usr/bin/env python3
"""Performs forward propagation"""
import tensorflow.compat.v1 as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Performs forward prop on x"""
    pred = create_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes)):
        pred = create_layer(pred, layer_sizes[i], activations[i])
    return pred
