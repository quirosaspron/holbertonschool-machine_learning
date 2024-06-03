#!/usr/bin/env python3
""" this projet is about keras
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, verbose=True, shuffle=False):
    """
    this is the task 5, train a model adding a validation data
    """

    if validation_data is None:
        history = network.fit(data, labels, epochs=epochs,
                              batch_size=batch_size,
                              verbose=False, shuffle=shuffle,
                              validation_data=validation_data)

    else:
        history = network.fit(data, labels,
                              epochs=epochs, batch_size=batch_size,
                              verbose=verbose, shuffle=shuffle,
                              validation_data=validation_data)
    return history
