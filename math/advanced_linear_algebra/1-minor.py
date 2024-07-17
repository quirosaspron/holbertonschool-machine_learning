#!/usr/bin/env python3
"""Gets the minor matrix of a matrix"""


def minor(matrix):
    """returns the minor matrix of a matrix"""

    if not isinstance(matrix, list):
        raise TypeError('matrix must be a list of lists')

    size = len(matrix)

    if size == 0:
        raise ValueError('matrix must be a list of lists')

    for row in matrix:
        if not isinstance(row, list):
            raise TypeError('matrix must be a list of lists')
        if len(row) != size:
            raise ValueError('matrix must be a non-empty square matrix')

    if size == 1:
        return [[1]]

    minor_matrix = [[0 for element in row] for row in matrix]

    for i in range(size):
        for j in range(size):
            minor_matrix[i][j] = determinant(get_minor(matrix, i, j))

    return minor_matrix


def determinant(matrix):
    """Gets the determinant of a matrix recursively"""

    if matrix == [[]]:
        return 1

    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError('matrix must be a list of lists')

    size = len(matrix)

    for row in matrix:
        if not isinstance(row, list):
            raise TypeError('matrix must be a list of lists')
        if len(row) != size:
            raise ValueError('matrix must be a square matrix')

    if size == 1:
        return matrix[0][0]

    if size == 2:
        det = matrix[0][0] * matrix[1][1]\
                       - matrix[0][1] * matrix[1][0]
        return det

    else:
        det = 0
        for i in range(size):
            minor = [row[:i] + row[i+1:] for row in (matrix[1:])]
            det += ((-1)**(i)) * matrix[0][i] * determinant(minor)

    return det


def get_minor(matrix, i, j):
    """Removes row i and column j from matrix"""
    return [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
