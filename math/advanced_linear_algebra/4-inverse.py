#!/usr/bin/env python3
"""Calculates the inverse matrix of a matrix"""


def inverse(matrix):
    """Returns the inverse matrix of a matrix"""

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

    adj = adjugate(matrix)
    det = determinant(matrix)

    if det == 0:
        return None

    inverse = [[e * (1/det) for e in row] for row in adj]

    return inverse


def adjugate(matrix):
    """Returns the adjugate matrix of a matrix"""

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

    cofactor_matrix = cofactor(matrix)
    adjugate_matrix = [[0 for element in row] for row in matrix]

    for i in range(size):
        for j in range(size):
            adjugate_matrix[i][j] = cofactor_matrix[j][i]

    return adjugate_matrix


def cofactor(matrix):
    """Returns the cofactor matrix of a matrix"""
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

    minor_matrix = minor(matrix)

    for i in range(size):
        for j in range(size):
            minor_matrix[i][j] = ((-1)**(i+j)) * minor_matrix[i][j]

    return minor_matrix


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
