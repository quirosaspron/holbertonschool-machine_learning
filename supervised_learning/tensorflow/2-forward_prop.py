#!/usr/bin/env python3
"""Performs forward propagation"""
import tensorflow.compat.v1 as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Performs forward prop on x"""
    for i in range(len(layer_sizes)):
        if i == 0:
            pred = create_layer(x, layer_sizes[i], activations[i])
        else:
            pred = create_layer(pred, layer_sizes[i], activations[i])
    return pred
