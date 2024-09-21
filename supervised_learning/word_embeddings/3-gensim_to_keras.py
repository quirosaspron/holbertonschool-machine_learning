#!/usr/bin/env python3
"""Converts a gensim word2vec model to a keras Embedding layer"""
from tensorflow.keras.layers import Embedding


def gensim_to_keras(model):
    """
        model: trained gensim word2vec model
        Returns: the trainable keras Embedding
    """
    # structure holding the result of training
    keyed_vectors = model.wv

    # vectors themselves, a 2D numpy array
    weights = keyed_vectors.vectors

    # which row in `weights` corresponds to which word?
    index_to_key = keyed_vectors.index_to_key

    layer = Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
    )
    return layer
