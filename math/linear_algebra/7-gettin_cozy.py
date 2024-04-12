"""Concatenates a matrix in certain axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenate a matrix in axis 1 or axis 2"""
    if len(mat1[0]) == len(mat2[0]) and axis == 0:
        return mat1 + mat2
    if len(mat1)== len(mat2) and axis == 1:
        new_mat = [row[:] for row in mat1]
        i = 0
        for row in mat2:
            for element in row:
                new_mat[i].append(element)
                i += 1
        return new_mat
    else:
        return None
