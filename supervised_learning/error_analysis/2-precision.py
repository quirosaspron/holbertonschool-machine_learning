#!/usr/bin/env python3
"""Calculates precision for each class in a confusion matrix"""
import numpy as np


def precision(confusion):
    """Returns an array containing the precision of each class"""
    confusion_matrix = confusion
    precision = np.zeros((confusion_matrix.shape[1],))
    for c in range(confusion_matrix.shape[1]):
        true_positives = 0
        false_positives = 0
        for r in range(confusion_matrix.shape[0]):
            if c == r:
                true_positives = confusion_matrix[r][c]
            else:
                false_positives += confusion_matrix[r][c]
        precision[c] = true_positives / (true_positives + false_positives)
    return precision
