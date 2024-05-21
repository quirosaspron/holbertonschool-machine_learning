#!/usr/bin/env python3
"""Calculates sensitivity for each class in a confusion matrix"""
import numpy as np


def sensitivity(confusion):
    """Returns an array containing the sensitivity of each class"""
    confusion_matrix = confusion
    sensitivity = np.zeros((confusion_matrix.shape[1],))
    for r in range(confusion_matrix.shape[0]):
        true_positives = 0
        false_negatives = 0
        for c in range(confusion_matrix.shape[1]):
            if r == c:
                true_positives = confusion_matrix[r][c]
            else:
                false_negatives += confusion_matrix[r][c]
        sensitivity[r] = true_positives / (true_positives + false_negatives)
    return sensitivity
