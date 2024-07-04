#!/usr/bin/env python3
"""Neural Style Transfer project by Mateo Quirós Asprón"""
import numpy as np
import tensorflow as tf


class NST:
    """Neural style transfer class"""
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initializes the class"""
        if not isinstance(style_image, np.ndarray):
            raise TypeError("style_image must be a numpy.ndarray \
with shape (h, w, 3)")

        if style_image.shape[-1] != 3:
            raise TypeError("style_image must be a numpy.ndarray \
with shape (h, w, 3)")

        if not isinstance(content_image, np.ndarray):
            raise TypeError("content_image must be a numpy.ndarray \
with shape (h, w, 3)")

        if content_image.shape[-1] != 3:
            raise TypeError("content_image must be a numpy.ndarray \
with shape (h, w, 3)")

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """Rescales the image"""
        height, width = image.shape[:2]
        if not isinstance(image, np.ndarray) or image.shape[-1] != 3:
            raise TypeError("image must be a numpy.ndarray \
with shape (h, w, 3)")
        max_side = np.maximum(height, width)
        scaling_factor = 512/max_side
        new_height = int(height * scaling_factor)
        new_width = int(width * scaling_factor)
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.expand_dims(image, axis=0)
        rescaled_image = tf.image.resize(image, (new_height,
                                         new_width), method='bicubic')
        rescaled_image = rescaled_image / 255.0
        rescaled_image = tf.clip_by_value(rescaled_image, 0, 1)
        return rescaled_image
