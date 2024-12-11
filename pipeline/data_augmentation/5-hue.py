#!/usr/bin/env python3
""" changes the hue of an image """
import tensorflow as tf


def change_hue(image, delta):
    """
    image: tensor containing the image to change
    delta is the amount the hue should change
    Returns the altered image
    """
    new_img = tf.image.adjust_hue(
                  image, delta)
    return new_img
