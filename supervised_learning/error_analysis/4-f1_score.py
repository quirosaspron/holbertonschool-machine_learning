#!/usr/bin/env python3
"""Calculates F1 score for each class in a confusion matrix"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Calculates the F1 score for each class in a confusion matrix"""
    classes = confusion.shape[0]
    f1_scores = np.zeros(classes)
    for c in range(classes):
        TP = confusion[c, c]
        FP = np.sum(confusion[:, c]) - TP
        FN = np.sum(confusion[c, :]) - TP
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        if precision + recall > 0:
            f1_scores[c] = 2 * precision * recall / (precision + recall)
        else:
            f1_scores[c] = 0
    return f1_scores
