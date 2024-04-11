#!/usr/bin/env python3
"""Transposes a matrix"""


def matrix_transpose(matrix):
    """ This will transpose a matrix"""
    columns = len(matrix[0])
    new_matrix = [[] for _ in range(columns)]
    for row in matrix:
        count = 0
        for element in row:
            new_matrix[count].append(element)
            count += 1
    return new_matrix
