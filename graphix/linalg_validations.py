from typing import Union

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


def truncate(s: str, max_length: int = 80, ellipsis: str = "...") -> str:
    "Auxilliary function to truncate a long string for formatting error messages."
    if len(s) <= max_length:
        return s
    return s[: max_length - len(ellipsis)] + ellipsis


def check_psd(matrix: np.ndarray, tol: float = 1e-15) -> bool:
    """
    check if a density matrix is positive semidefinite by diagonalizing.

    Parameters
    ----------
    matrix : np.ndarray
        matrix to check
    tol : float
        tolerance on the small negatives. Default 1e-15.
    """

    evals = np.linalg.eigvalsh(matrix)

    if not all(evals >= -tol):
        raise ValueError("The matrix {truncate(str(matrix))} is not positive semi-definite.")

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


def check_data_normalization(data: Union[list, tuple, np.ndarray]) -> bool:
    # NOTE use np.conjugate() instead of object.conj() to certify behaviour when using non-numpy float/complex types
    opsu = np.array([i["coef"] * np.conj(i["coef"]) * i["operator"].conj().T @ i["operator"] for i in data])

    if not np.allclose(np.sum(opsu, axis=0), np.eye(2 ** int(np.log2(len(data[0]["operator"]))))):
        raise ValueError(f"The specified channel is not normalized {np.sum(opsu, axis=0)}.")
    return True


def check_data_dims(data: Union[list, tuple, np.ndarray]) -> bool:
    # convert to set to remove duplicates
    dims = set([i["operator"].shape for i in data])

    # check all the same dimensions and that they are square matrices
    # TODO replace by using array.ndim
    if len(dims) != 1:
        raise ValueError(f"All provided Kraus operators do not have the same dimension {dims}!")

    assert check_square(data[0]["operator"])

    return True


def check_data_values_type(data: Union[list, tuple, np.ndarray]) -> bool:
    if not all(
        isinstance(i, dict) for i in data
    ):  # ni liste ni ensemble mais iterable (lazy) pas stocké, executé au besoin
        raise TypeError("All values are not dictionaries.")

    if not all(set(i.keys()) == {"coef", "operator"} for i in data):
        raise KeyError("The keys of the indivudal Kraus operators must be coef and operator.")

    if not all(isinstance(i["operator"], np.ndarray) for i in data):
        raise TypeError("All operators don't have the same type and must be np.ndarray.")

    for i in data:
        if not i["operator"].dtype in (int, float, complex, np.float64, np.complex128):
            raise TypeError(f"All operators dtype must be scalar and not {i['operator'].dtype}.")

    if not all(isinstance(i["coef"], (int, float, complex, np.float64, np.complex128)) for i in data):
        raise TypeError("All coefs dtype must be scalar.")

    return True


def check_rank(data: Union[list, tuple, np.ndarray]) -> bool:
    # already checked that the data is list of square matrices
    if len(data) > data[0]["operator"].shape[0] ** 2:
        raise ValueError(
            f"Incorrect number of Kraus operators in the expansion. This number must be an integer between 1 and the dimension squared."
        )

    return True
