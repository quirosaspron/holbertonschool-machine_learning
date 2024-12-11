#!/usr/bin/env python3
""" randomly changes the brightness of an image """
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    image: tensor containing the image to change
    max_delta: max amount the image should be brightened (or darkened)
    Returns the altered image
    """
    new_img = tf.image.random_brightness(
                  image, max_delta)
    return new_img
