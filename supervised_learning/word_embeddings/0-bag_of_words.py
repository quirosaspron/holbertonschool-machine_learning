#!/usr/bin/env python3
"""Creates a bag of words embedding matrix"""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
        sentences: list of sentences
        vocab: list of the vocabulary words for the analysis
        returns embeddings, features
        embeddings: freq of features in sentences
        features: unique vocab words in sentences

    """
    sentences = [sentence.lower() for sentence in sentences]
    len_s = len(sentences)
    features = []

    # Replace possessive forms with their non-possessive versions
    sentences = [re.sub(r"(\b\w+)'s\b", r'\1', sentence)
                 for sentence in sentences]

    # Remove non-word characters
    sentences = [re.sub(r"[^\w\s]", ' ', sentence) for sentence in sentences]

    # Get features
    if vocab is not None:
        for sentence in sentences:
            for word in vocab:
                if word in sentence.split() and word not in features:
                    features.append(word)

    else:
        for sentence in sentences:
            for word in sentence.split():
                if word not in features:
                    features.append(word)
        features = sorted(features)

    len_f = len(features)
    embeddings = np.zeros((len_s, len_f), int)

    # Get frequency of each feature in each sentence
    for i, sentence in enumerate(sentences):
        for j, word in enumerate(features):
            embeddings[i][j] += sentence.split().count(word)

    return embeddings, np.array(features)
