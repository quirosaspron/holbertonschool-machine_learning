#!/usr/bin/env python3
""" this projet is about keras
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, verbose=True, shuffle=False):
    """
    this is the task 7, train a model and add
    inverse time decay, using "K.callbacks.LearningRateScheduler"
    """

    if validation_data is None:
        history = network.fit(data, labels, epochs=epochs,
                              batch_size=batch_size,
                              verbose=False, shuffle=shuffle,
                              validation_data=validation_data)

    else:

        # Define your learning rate schedule function with inverse time decay
        def lr_schedule(epoch):
            """
            Inverse Time Decay Learning Rate Schedule
            """
            learning_rate = alpha / (1 + decay_rate * epoch)

            return learning_rate

        # callback early Stopping when back propgation is null
        callback_early = K.callbacks.EarlyStopping(patience=patience)
        # callback decay inverse time
        callback_invertime_decay = K.callbacks.\
            LearningRateScheduler(lr_schedule, verbose=True)

        history = network.fit(data, labels,
                              epochs=epochs, batch_size=batch_size,
                              verbose=verbose, shuffle=shuffle,
                              validation_data=validation_data,
                              callbacks=[callback_early,
                                         callback_invertime_decay])

    return history
