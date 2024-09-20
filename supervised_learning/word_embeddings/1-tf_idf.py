#!/usr/bin/env python3
"""
Defines function that creates a TF-IDF embedding
"""


from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """ Creates a TF-IDF embedding """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    x = vectorizer.fit_transform(sentences)
    embeddings = x.toarray()
    features = vectorizer.get_feature_names_out()
    return embeddings, features
