#!/usr/bin/env python3
"""Creates a bag of words embedding matrix"""
import numpy as np


def bag_of_words(sentences, vocab=None):
    """
        sentences: list of sentences
        vocab: list of the vocabulary words for the analysis
        returns embeddings, features
        embeddings: freq of features in sentences
        features: unique vocab words in sentences

    """
    sentences = sentences
    len_s = len(sentences)
    features = []

    # Get features
    if vocab is not None:
        for sentence in sentences:
            for word in vocab:
                if word in sentence.lower().split():
                    features.append(word)

    else:
        for sentence in sentences:
            for word in sentence.lower().split():
                if word not in features:
                    features.append(word)

    features = sorted(features)
    len_f = len(features)
    embeddings = np.zeros((len_s, len_f), int)

    # Get frequency of each feature in each sentence
    for i, sentence in enumerate(sentences):
        for j, word in enumerate(features):
            embeddings[i][j] += sentence.lower().split().count(word)

    return embeddings, features
