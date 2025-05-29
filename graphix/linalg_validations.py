"""Validation functions for linear algebra."""

from __future__ import annotations

from typing import TypeVar

import numpy as np
import numpy.typing as npt

_T = TypeVar("_T", bound=np.generic)


def is_square(matrix: npt.NDArray[_T]) -> bool:
    """Check if matrix is square."""
    if matrix.ndim != 2:
        return False
    rows, cols = matrix.shape
    # Circumvent a regression in numpy 2.1.
    # Note that this regression is already fixed in numpy 2.2.
    # reveal_type(rows) -> Any
    # reveal_type(cols) -> Any
    assert isinstance(rows, int)
    assert isinstance(cols, int)
    return rows == cols


def is_qubitop(matrix: npt.NDArray[_T]) -> bool:
    """Check if matrix is a square matrix with a power of 2 dimension."""
    if not is_square(matrix):
        return False
    size, _ = matrix.shape
    # Circumvent a regression in numpy 2.1.
    # Note that this regression is already fixed in numpy 2.2.
    # reveal_type(size) -> Any
    assert isinstance(size, int)
    return size > 0 and size & (size - 1) == 0


def is_hermitian(matrix: npt.NDArray[_T]) -> bool:
    """Check if matrix is hermitian."""
    if not is_square(matrix):
        return False
    return np.allclose(matrix, matrix.transpose().conjugate())


def is_psd(matrix: npt.NDArray[_T], tol: float = 1e-15) -> bool:
    """
    Check if a density matrix is positive semidefinite by diagonalizing.

    Parameters
    ----------
    matrix : np.ndarray
        matrix to check
    tol : float
        tolerance on the small negatives. Default 1e-15.
    """
    if not is_square(matrix):
        return False
    if tol < 0:
        raise ValueError("tol must be non-negative.")
    if not is_hermitian(matrix):
        return False
    evals = np.linalg.eigvalsh(matrix.astype(np.complex128))
    return all(evals >= -tol)


def is_unit_trace(matrix: npt.NDArray[_T]) -> bool:
    """Check if matrix has trace 1."""
    if not is_square(matrix):
        return False
    return np.allclose(matrix.trace(), 1.0)
