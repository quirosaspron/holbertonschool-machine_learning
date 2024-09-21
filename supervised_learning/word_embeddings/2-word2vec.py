#!/usr/bin/env python3
"""Creates, builds and trains a gensim word2vec model"""
from gensim.models import Word2Vec


def word2vec_model(sentences, vector_size=100, min_count=5,
                   window=5, negative=5, cbow=True, epochs=5,
                   seed=0, workers=1):
    """sentences: list of sentences to be trained on
       vector_size: dimensionality of embedding layer
       min_count: minimum number of occurrences of a
                  word for use in training
       window: maximum distance between current and
               predicted word within a sentence
       negative: size of negative sampling
       cbow: True for CBOW; False for Skip-gram
       epochs: number of iterations to train over
       seed: seed for the random number generator
       workers: number of worker threads to train the model
       Returns: the trained model """

    sg = 0 if cbow else 1

    model = Word2Vec(sentences=sentences,
                     vector_size=vector_size,
                     seed=seed,
                     sg=sg,
                     negative=negative,
                     window=window,
                     min_count=min_count,
                     workers=workers)

    model.train(sentences,
                total_examples=model.corpus_count,
                epochs=epochs)

    return model
