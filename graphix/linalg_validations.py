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
