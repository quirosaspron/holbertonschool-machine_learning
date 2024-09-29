#!/usr/bin/env python3
""" Builds an RNN encoder class """
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """ RNN Encoder class """
    def __init__(self, vocab, embedding, units, batch_size):
        """ Class constructor """
        super(RNNEncoder, self).__init__()  # Initialize parent class
        self.units = units
        self.batch = batch_size
        # Embeding layer
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        # RNN weights initializer
        initializer = tf.keras.initializers.GlorotUniform()
        # RNN layer
        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       kernel_initializer=initializer)

    def initialize_hidden_state(self):
        """ Initializes the hidden states
            for the RNN cell to a tensor
            of zeros """
        return tf.zeros((self.batch, self.units))

    def call(self, x, hidden):
        """ Returns the outputs and the
            last hidden state of the encoder """
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state
