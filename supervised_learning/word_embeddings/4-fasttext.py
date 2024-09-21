#!/usr/bin/env python3
""" creates, builds and trains a genism fastText model"""
import gensim


def fasttext_model(sentences, vector_size=100,
                   min_count=5, negative=5,
                   window=5, cbow=True, epochs=5,
                   seed=0, workers=1):
    """
       sentences: list of sentences to be trained on
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
       Returns: the trained model
     """

    sg = 0 if cbow else 1

    model = gensim.models.FastText(sentences=sentences,
                                   vector_size=vector_size,
                                   seed=seed,
                                   sg=sg,
                                   negative=negative,
                                   window=window,
                                   min_count=min_count,
                                   workers=workers)

    # build the vocabulary
    # this hepls to create the one-hot encoding of the words
    model.build_vocab(sentences)

    model.train(sentences,
                total_examples=model.corpus_count,
                epochs=epochs)

    return model
