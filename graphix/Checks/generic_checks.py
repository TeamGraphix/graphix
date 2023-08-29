import numpy as np


def check_square(matrix: np.ndarray) -> bool:
    """
    check if matrix is a square matrix with a power of 2 dimension.
    """

    if len(matrix.shape) != 2:
        raise ValueError(f"The object has {len(matrix.shape)} axes but must have 2 to be a matrix.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Matrix must be square but has different dimensions {matrix.shape}.")
    size = matrix.shape[0]
    if size & (size - 1) != 0:
        raise ValueError(f"Matrix size must be a power of two but is {size}.")
    return True


def check_hermitian(matrix: np.ndarray) -> bool:
    """
    check if matrix is hermitian. After check_square.
    """

    if not np.allclose(matrix, matrix.transpose().conjugate()):
        raise ValueError("The matrix is not Hermitian.")
    return True


def check_unit_trace(matrix: np.ndarray) -> bool:
    """
    check if matrix has trace 1. After check_square.
    """

    if not np.allclose(matrix.trace(), 1.0):
        raise ValueError("The matrix does not have unit trace.")
    return True
