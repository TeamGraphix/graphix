"""Validation functions for linear algebra."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from typing_extensions import Concatenate, ParamSpec

if TYPE_CHECKING:
    from collections.abc import Callable


def is_square(matrix: npt.NDArray) -> bool:
    """Check if matrix is square."""
    if matrix.ndim != 2:
        return False
    rows, cols = matrix.shape
    return rows == cols


_P = ParamSpec("_P")


def _is_square_deco(f: Callable[Concatenate[npt.NDArray, _P], bool]) -> Callable[Concatenate[npt.NDArray, _P], bool]:
    """Check if matrix is square, and then call the function."""

    @functools.wraps(f)
    def _f(matrix: npt.NDArray, *args: _P.args, **kwargs: _P.kwargs) -> bool:
        if not is_square(matrix):
            raise ValueError("Need to ensure that matrix is square first.")
        return f(matrix, *args, **kwargs)

    return _f


@_is_square_deco
def is_qubitop(matrix: npt.NDArray) -> bool:
    """Check if matrix is a square matrix with a power of 2 dimension."""
    size, _ = matrix.shape
    return size > 0 and size & (size - 1) == 0


@_is_square_deco
def is_psd(matrix: npt.NDArray, tol: float = 1e-15) -> bool:
    """
    Check if a density matrix is positive semidefinite by diagonalizing.

    Parameters
    ----------
    matrix : np.ndarray
        matrix to check
    tol : float
        tolerance on the small negatives. Default 1e-15.
    """
    if tol < 0:
        raise ValueError("tol must be non-negative.")
    evals = np.linalg.eigvalsh(matrix)
    return all(evals >= -tol)


@_is_square_deco
def is_hermitian(matrix: npt.NDArray) -> bool:
    """Check if matrix is hermitian."""
    return np.allclose(matrix, matrix.transpose().conjugate())


@_is_square_deco
def is_unit_trace(matrix: npt.NDArray) -> bool:
    """Check if matrix has trace 1."""
    if not np.allclose(matrix.trace(), 1.0):
        return False
    return True


def check_data_normalization(data: list | tuple | np.ndarray) -> bool:
    """Check that data is normalized."""
    # NOTE use np.conjugate() instead of object.conj() to certify behaviour when using non-numpy float/complex types
    opsu = np.array([i["coef"] * np.conj(i["coef"]) * i["operator"].conj().T @ i["operator"] for i in data])

    if not np.allclose(np.sum(opsu, axis=0), np.eye(2 ** int(np.log2(len(data[0]["operator"]))))):
        raise ValueError(f"The specified channel is not normalized {np.sum(opsu, axis=0)}.")
    return True


def check_data_dims(data: list | tuple | np.ndarray) -> bool:
    """Check that of Kraus operators have the same dimension."""
    # convert to set to remove duplicates
    dims = set([i["operator"].shape for i in data])

    # check all the same dimensions and that they are square matrices
    # TODO replace by using array.ndim
    if len(dims) != 1:
        raise ValueError(f"All provided Kraus operators do not have the same dimension {dims}!")

    assert is_square(data[0]["operator"])

    return True


def check_data_values_type(data: list | tuple | np.ndarray) -> bool:
    """Check the types of Kraus operators."""
    if not all(
        isinstance(i, dict) for i in data
    ):  # ni liste ni ensemble mais iterable (lazy) pas stocké, executé au besoin
        raise TypeError("All values are not dictionaries.")

    if not all(set(i.keys()) == {"coef", "operator"} for i in data):
        raise KeyError("The keys of the indivudal Kraus operators must be coef and operator.")

    if not all(isinstance(i["operator"], np.ndarray) for i in data):
        raise TypeError("All operators don't have the same type and must be np.ndarray.")

    for i in data:
        if i["operator"].dtype not in (int, float, complex, np.float64, np.complex128):
            raise TypeError(f"All operators dtype must be scalar and not {i['operator'].dtype}.")

    if not all(isinstance(i["coef"], (int, float, complex, np.float64, np.complex128)) for i in data):
        raise TypeError("All coefs dtype must be scalar.")

    return True


def check_rank(data: list | tuple | np.ndarray) -> bool:
    """Check the rank of Kraus operators."""
    # already checked that the data is list of square matrices
    if len(data) > data[0]["operator"].shape[0] ** 2:
        raise ValueError(
            "Incorrect number of Kraus operators in the expansion. This number must be an integer between 1 and the dimension squared."
        )

    return True
