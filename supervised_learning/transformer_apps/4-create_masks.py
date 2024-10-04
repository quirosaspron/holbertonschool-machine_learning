#!/usr/bin/env python3
"""Generates the masks for training and validation"""
import tensorflow as tf


def create_padding_mask(seq):
    """Creates a padding mask for a given sequence"""
    # Generate mask for padding tokens
    # (value 0 is considered padding)
    padding_mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # Shape: (batch_size, 1, 1, seq_len)
    return padding_mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    """Creates a look-ahead mask for a sequence"""
    # Create a lower triangular matrix
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    # Shape: (size, size)
    return look_ahead_mask


def create_masks(inputs, target):
    """
    Creates all the masks for training/validation
    in the Transformer model.

    Returns:
        encoder_mask: Padding mask for the encoder of
                      shape (batch_size, 1, 1, seq_len_in)

        combined_mask: Mask for the decoder's first attention
                       block of shape (batch_size, 1,
                       seq_len_out, seq_len_out)

        decoder_mask: Padding mask for the encoder outputs of shape
                      (batch_size, 1, 1, seq_len_in).
    """
    # Encoder padding mask (to mask padding tokens in the input)
    encoder_mask = create_padding_mask(inputs)

    # Decoder padding mask (to mask padding tokens in the target)
    decoder_padding_mask = create_padding_mask(inputs)

    # Look-ahead mask for the target sequence to mask future tokens
    look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])

    # Decoder target padding mask (to mask padding tokens in the target)
    decoder_target_padding_mask = create_padding_mask(target)

    # Combined mask
    # (maximum of look-ahead mask and decoder target padding mask)
    combined_mask = tf.maximum(decoder_target_padding_mask[:, :, :, :],
                               look_ahead_mask[tf.newaxis, tf.newaxis, :, :])

    return encoder_mask, combined_mask, decoder_padding_mask
