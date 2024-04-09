#!/usr/bin/env python3
def matrix_transpose(matrix):
    new_matrix = [[] for row in matrix]
    for row in matrix:
        count = 0
        for element in row:
            new_matrix[count].append(element)
            count += 1
    return new_matrix
