#!/usr/bin/env python3
def add_matrices2D(mat1, mat2):
    matrix_shape = __import__('2-size_me_please').matrix_shape
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    else:
        new_matrix = [[] for row in mat1]
        r = 0
        for row in mat1:
            i = 0
            for element in row:
                new_matrix[r].append(mat1[r][i]+mat2[r][i])
                i += 1
            r += 1
        return new_matrix
