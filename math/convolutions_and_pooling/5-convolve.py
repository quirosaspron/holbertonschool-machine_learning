#!/usr/bin/env python3
"""
Performs a convolution on images using multiple kernels
"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using mulitple kernels.
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph = 0
        pw = 0

    # Calculate output dimensions
    output_h = (h + 2 * ph - kh) // sh + 1
    output_w = (w + 2 * pw - kw) // sw + 1

    # Pad the images with zeros
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant')

    # Initialize convolution output array
    convolved = np.zeros((m, output_h, output_w, nc))

    for k in range(nc):
        kernel = kernels[:, :, :, k]
        for i in range(output_h):
            for j in range(output_w):
                # Extract region from padded images, scaling indexes by stride
                region = padded_images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
                # Convolve each image (m) for this region (i, j, k)
                convolved[:, i, j, k] = np.sum(region * kernel, axis=(1, 2, 3))

    return convolved
