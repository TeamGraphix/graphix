import numpy as np


# TODO separate the check for power of 2 dimension from the rest
# TODO investigate numpy.typing for ndarray NDArray or ArrayLike (can be converted to arrays)
# https://stackoverflow.com/questions/35673895/type-hinting-annotation-pep-484-for-numpy-ndarray
# https://numpy.org/doc/stable/reference/typing.html#module-numpy.typing


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


def check_psd(matrix: np.ndarray, tol: float = 1e-15) -> bool:
    """
    check if a density matrix is positive semidefinite by diagonalizing.
    After check_square and check_hermitian (osef) so that it already is square with power of 2 dimension.


    Parameters
    ----------
    matrix : np.ndarray
        matrix to check. Normally already square and 2**n x 2**n
    tol : float
        tolerance on the small negatives. Default 1e-15.
    method : 'choleski' or 'sylvester' (or 'eigendecomp'). Default 'choleski'
        if method = 'choleski': attempts a Choleski decomposition. If the matrix is not PSD raises a numpy.linalg.LinAlgError. (but error can be something else...)
        if method = 'sylvester': computes the determinants of all square matrices of increasing dimension starting from the top left corner. Sylvester : PSD <-> all principal minors are non negative. NOT just leading!!!!!!
        NOPE. Sylvester for PSD is all principal minors, not just leading !!! So VERY expensive.
    """

    # if matrix is not hermitian, raises an error
    # if no error: eigenvals are real

    evals = np.linalg.eigvalsh(matrix)
    print(matrix)
    print(evals)
    # remove small negatives like -1e-17
    evals[np.abs(evals) < tol] = 0

    # sort (ascending order)
    evals.sort()

    # PSD test. Just look
    if not evals[0] >= 0:
        raise ValueError("The matrix is not positive semi-definite.")

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
