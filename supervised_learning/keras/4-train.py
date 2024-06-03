#!/usr/bin/env python3
""" this projet is about keras
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """
    this is the task 4, train a model
    """

    history = network.fit(data, labels, epochs=epochs, batch_size=batch_size,
                          verbose=verbose, shuffle=shuffle)

    if verbose is False:
        valuesloss = [float("{:.5f}".format(x))
                      for x in history.history["loss"]]
        valueaccurate = [float("{:.5f}".format(x))
                         for x in history.history["accuracy"]]

    return history
