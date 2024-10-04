#!/usr/bin/env python3
"""Loads and preps a dataset for machine translation"""
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """Dataset class"""
    def __init__(self):
        """Class constructor"""
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train',
            as_supervised=True)

        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation',
            as_supervised=True)

        self.tokenizer_pt, self.tokenizer_en = (
            self.tokenize_dataset(self.data_train))

    def tokenize_dataset(self, data):
        """ Creates sub-word tokenizers

            - data: tf.data.Dataset whose examples are
              formatted as a tuple (pt, en)

            - pt: tf.Tensor containing
              the Portuguese sentence

            - en is the tf.Tensor containing the
              corresponding English sentence """

        # Load the pre-trained Portuguese tokenizer
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased')

        # Load the pre-trained English tokenizer
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased')

        # Decode the sentences
        en_sentences = (en.numpy().decode('utf-8') for pt, en in data)
        pt_sentences = (pt.numpy().decode('utf-8') for pt, en in data)

        # Train tokenizers with a max vocab size of 2**13
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_sentences, 2**13)
        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_sentences, 2**13)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encodes a translation into tokens"""
        # Decode pt and en
        pt = pt.numpy().decode('utf-8')
        en = en.numpy().decode('utf-8')

        # Tokenize them
        pt_tokens = [2**13] + self.tokenizer_pt.encode(
            pt, add_special_tokens=False) + [2**13 + 1]
        en_tokens = [2**13] + self.tokenizer_en.encode(
            en, add_special_tokens=False) + [2**13 + 1]

        return pt_tokens, en_tokens
