#!/usr/bin/env python3
""" randomly adjusts the contrast of an image """
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    image: 3D Tensor representing the input image to adjust the contrast
    lower: lower bound of the random contrast factor range
    upper: upper bound of the random contrast factor range
    Returns the contrast-adjusted image
    """
    new_img = tf.image.random_contrast(
                  image, lower, upper)
    return new_img
