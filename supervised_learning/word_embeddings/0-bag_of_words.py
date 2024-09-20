#!/usr/bin/env python3
"""Creates a bag of words embedding matrix"""
import numpy as np


def clean_sentence(sentence):
    """ Makes the sentence lowercase and removes non alphabet
        characters"""
    sentence = sentence.lower()
    alphabet = set("abcdefghijklmnopqrstuvwxyz")
    cleaned = []

    for char in sentence:
        if char in alphabet or char.isspace():
            cleaned.append(char)

    cleaned_sentence = ''
    for char in cleaned:
        cleaned_sentence += char

    return cleaned_sentence


def bag_of_words(sentences, vocab=None):
    """
        sentences: list of sentences
        vocab: list of the vocabulary words for the analysis
        returns embeddings, features
        embeddings: freq of features in sentences
        features: unique vocab words in sentences

    """
    sentences = [clean_sentence(sentence) for sentence in sentences]
    len_s = len(sentences)
    features = []

    # Get features
    if vocab is not None:
        for sentence in sentences:
            for word in vocab:
                if word in sentence.split():
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

    return embeddings, features
