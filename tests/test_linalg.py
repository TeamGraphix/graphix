from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pytest

from graphix._linalg import MatGF2, solve_f2_linear_system

if TYPE_CHECKING:
    from numpy.random import Generator
    from pytest_benchmark import BenchmarkFixture


class LinalgTestCase(NamedTuple):
    matrix: MatGF2
    rank: int
    kernel_dim: int
    right_invertible: bool


class LSF2TestCase(NamedTuple):
    mat: MatGF2
    b: MatGF2


def prepare_test_matrix() -> list[LinalgTestCase]:
    return [
        # empty matrix
        LinalgTestCase(
            MatGF2(np.array([[]], dtype=np.uint8)),
            0,
            0,
            False,
        ),
        # column vector
        LinalgTestCase(
            MatGF2(np.array([[1], [1], [1]], dtype=np.uint8)),
            1,
            0,
            False,
        ),
        # row vector
        LinalgTestCase(
            MatGF2(np.array([[1, 1, 1]], dtype=np.uint8)),
            1,
            2,
            True,
        ),
        # diagonal matrix
        LinalgTestCase(
            MatGF2(np.diag(np.ones(10)).astype(int)),
            10,
            0,
            True,
        ),
        # full rank dense matrix
        LinalgTestCase(
            MatGF2(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)),
            3,
            0,
            True,
        ),
        # not full-rank matrix
        LinalgTestCase(
            MatGF2(np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=np.uint8)),
            2,
            1,
            False,
        ),
        # non-square matrix
        LinalgTestCase(
            MatGF2(np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)),
            2,
            1,
            True,
        ),
        # non-square matrix
        LinalgTestCase(
            MatGF2(np.array([[1, 0], [0, 1], [1, 0]], dtype=np.uint8)),
            2,
            0,
            False,
        ),
    ]


def prepare_test_f2_linear_system() -> list[LSF2TestCase]:
    test_cases: list[LSF2TestCase] = []

    # `mat` must be in row echelon form.
    # `b` must have zeros in the indices corresponding to the zero rows of `mat`.

    test_cases.extend(
        (
            LSF2TestCase(mat=MatGF2([[1, 0, 1, 1], [0, 1, 0, 1], [0, 0, 0, 1]]), b=MatGF2([1, 0, 0])),
            LSF2TestCase(
                mat=MatGF2([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]),
                b=MatGF2([0, 1, 1, 0, 0]),
            ),
            LSF2TestCase(
                mat=MatGF2([[1, 1, 1], [0, 0, 1], [0, 0, 0], [0, 0, 0]]),
                b=MatGF2([0, 0, 0, 0]),
            ),
            LSF2TestCase(
                mat=MatGF2([[1, 0, 0], [0, 1, 1], [0, 0, 1], [0, 0, 0]]),
                b=MatGF2([1, 0, 1, 0]),
            ),
            LSF2TestCase(
                mat=MatGF2([[1, 0, 1], [0, 1, 0], [0, 0, 1]]),
                b=MatGF2([1, 1, 1]),
            ),
        )
    )

    return test_cases


def verify_elimination(mat: MatGF2, mat_red: MatGF2, n_cols_red: int, full_reduce: bool) -> None:
    """Test gaussian elimination (GE).

    Parameters
    ----------
    mat : MatGF2
        Original matrix.
    mat_red : MatGF2
        Gaussian-eliminated matrix.
    n_cols_red : int
        Number of columns over which `mat` was reduced.
    full_reduce : bool
        Flag to check for row-reduced echelon form (`True`) or row echelon form (`False`).

    Notes
    -----
    It tests that:
        1) Matrix is in row echelon form (REF) or row-reduced echelon form.
        2) The procedure only entails row operations.

        Check (2) implies that the GE procedure can be represented by a linear transformation. Thefore, we perform GE on :math:`A = [M|1]`, with :math:`M` the test matrix and :math:`1` the identiy, and we verify that :math:`M = L^{-1}M'`, where :math:`M', L` are the left and right blocks of :math:`A` after gaussian elimination.
    """
    mat_red_block = MatGF2(mat_red[:, :n_cols_red])
    # Check 1
    p = -1  # pivot
    for i, row in enumerate(mat_red_block):
        col_idxs = np.flatnonzero(row)  # Column indices with 1s
        if col_idxs.size == 0:
            assert not mat_red_block[i:, :].any()  # If there aren't any 1s, we verify that the remaining rows are all 0
            break
        j = col_idxs[0]
        assert j > p
        if full_reduce:
            assert (
                np.sum(mat_red_block[:, j] == 1) == 1
            )  # If checking row-reduced echelon form, verify it is the only 1.
        p = j

    # Check 2
    ncols = mat.shape[1]
    mat_ge = MatGF2(mat_red[:, :ncols])
    mat_l = MatGF2(mat_red[:, ncols:])

    mat_linv = mat_l.right_inverse()
    if mat_linv is not None:
        assert np.all((mat_linv @ mat_ge) % 2 == mat)  # Test with numpy matrix product.


class TestLinAlg:
    @pytest.mark.parametrize("test_case", prepare_test_matrix())
    def test_compute_rank(self, test_case: LinalgTestCase) -> None:
        mat = test_case.matrix
        rank = test_case.rank
        assert mat.compute_rank() == rank

    @pytest.mark.parametrize("test_case", prepare_test_matrix())
    def test_right_inverse(self, benchmark: BenchmarkFixture, test_case: LinalgTestCase) -> None:
        mat = test_case.matrix
        rinv = benchmark(mat.right_inverse)

        if test_case.right_invertible:
            assert rinv is not None
            ident = MatGF2(np.eye(mat.shape[0], dtype=np.uint8))
            assert np.all((mat @ rinv) % 2 == ident)  # Test with numpy matrix product.
        else:
            assert rinv is None

    @pytest.mark.parametrize("test_case", prepare_test_matrix())
    def test_gauss_elimination(self, test_case: LinalgTestCase) -> None:
        mat = test_case.matrix
        mat_red = mat.gauss_elimination(ncols=mat.shape[1], copy=True)
        verify_elimination(mat, mat_red, mat.shape[1], full_reduce=False)

    @pytest.mark.parametrize("test_case", prepare_test_matrix())
    def test_null_space(self, benchmark: BenchmarkFixture, test_case: LinalgTestCase) -> None:
        mat = test_case.matrix
        kernel_dim = test_case.kernel_dim

        kernel = benchmark(mat.null_space)

        assert kernel_dim == kernel.shape[0]
        for v in kernel:
            p = (mat @ v.transpose()) % 2  # Test with numpy matrix product.
            assert ~p.any()

    @pytest.mark.parametrize("test_case", prepare_test_f2_linear_system())
    def test_solve_f2_linear_system(self, benchmark: BenchmarkFixture, test_case: LSF2TestCase) -> None:
        mat = test_case.mat
        b = test_case.b

        x = benchmark(solve_f2_linear_system, mat, b)

        assert np.all((mat @ x) % 2 == b)  # Test with numpy matrix product.

    def test_row_reduction(self, fx_rng: Generator) -> None:
        sizes = [(10, 10), (3, 7), (6, 2)]
        ncols = [4, 5, 2]

        for size, ncol in zip(sizes, ncols, strict=True):
            mat = MatGF2(fx_rng.integers(size=size, low=0, high=2, dtype=np.uint8))
            mat_red = mat.row_reduction(ncols=ncol, copy=True)
            verify_elimination(mat, mat_red, ncol, full_reduce=True)
