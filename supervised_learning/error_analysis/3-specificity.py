#!/usr/bin/env python3
"""Calculates specificity for each class in a confusion matrix"""
import numpy as np


def specificity(confusion):
    """Returns an array containing the specificity of each class"""
    classes = confusion.shape[0]
    specificity = np.zeros(classes)

    for r in range(classes):
        true_negatives = 0
        false_positives = 0
        for i in range(classes):
            for j in range(classes):
                if i != r and j != r:
                    true_negatives += confusion[i][j]
                if i != r and j == r:
                    false_positives += confusion[i][j]
        specificity[r] = true_negatives / (true_negatives + false_positives)
    
    return specificity
