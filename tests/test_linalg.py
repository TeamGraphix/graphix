from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import galois
import numpy as np
import pytest

from graphix.linalg import MatGF2, solve_f2_linear_system

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
            MatGF2(np.array([[]], dtype=np.int_)),
            0,
            0,
            False,
        ),
        # column vector
        LinalgTestCase(
            MatGF2(np.array([[1], [1], [1]], dtype=np.int_)),
            1,
            0,
            False,
        ),
        # row vector
        LinalgTestCase(
            MatGF2(np.array([[1, 1, 1]], dtype=np.int_)),
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
            MatGF2(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.int_)),
            3,
            0,
            True,
        ),
        # not full-rank matrix
        LinalgTestCase(
            MatGF2(np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=np.int_)),
            2,
            1,
            False,
        ),
        # non-square matrix
        LinalgTestCase(
            MatGF2(np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int_)),
            2,
            1,
            True,
        ),
        # non-square matrix
        LinalgTestCase(
            MatGF2(np.array([[1, 0], [0, 1], [1, 0]], dtype=np.int_)),
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


class TestLinAlg:
    def test_add_row(self) -> None:
        test_mat = MatGF2(np.diag(np.ones(2, dtype=np.int_)))
        test_mat.add_row()
        assert test_mat.data.shape == (3, 2)
        assert np.all(test_mat.data == np.array([[1, 0], [0, 1], [0, 0]]))

    def test_add_col(self) -> None:
        test_mat = MatGF2(np.diag(np.ones(2, dtype=np.int_)))
        test_mat.add_col()
        assert test_mat.data.shape == (2, 3)
        assert np.all(test_mat.data == galois.GF2(np.array([[1, 0, 0], [0, 1, 0]])))

    def test_remove_row(self) -> None:
        test_mat = MatGF2(np.array([[1, 0], [0, 1], [0, 0]], dtype=np.int_))
        test_mat.remove_row(2)
        assert np.all(test_mat.data == np.array([[1, 0], [0, 1]]))

    def test_remove_col(self) -> None:
        test_mat = MatGF2(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.int_))
        test_mat.remove_col(2)
        assert np.all(test_mat.data == np.array([[1, 0], [0, 1]]))

    def test_swap_row(self) -> None:
        test_mat = MatGF2(np.array([[1, 0], [0, 1], [0, 0]], dtype=np.int_))
        test_mat.swap_row(1, 2)
        assert np.all(test_mat.data == np.array([[1, 0], [0, 0], [0, 1]]))

    def test_swap_col(self) -> None:
        test_mat = MatGF2(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.int_))
        test_mat.swap_col(1, 2)
        assert np.all(test_mat.data == np.array([[1, 0, 0], [0, 0, 1]]))

    @pytest.mark.parametrize("test_case", prepare_test_matrix())
    def test_get_rank(self, test_case: LinalgTestCase) -> None:
        mat = test_case.matrix
        rank = test_case.rank
        assert mat.get_rank() == rank

    @pytest.mark.parametrize("test_case", prepare_test_matrix())
    def test_right_inverse(self, benchmark: BenchmarkFixture, test_case: LinalgTestCase) -> None:
        mat = test_case.matrix
        rinv = benchmark(mat.right_inverse)

        if test_case.right_invertible:
            assert rinv is not None
            ident = MatGF2(np.eye(mat.data.shape[0], dtype=np.int_))
            assert mat @ rinv == ident
        else:
            assert rinv is None

    @pytest.mark.parametrize("test_case", prepare_test_matrix())
    def test_gaussian_elimination(self, test_case: LinalgTestCase) -> None:
        """Test gaussian elimination (GE).

        It tests that:
            1) Matrix is in row echelon form (REF).
            2) The procedure only entails row operations.

        Check (2) implies that the GE procedure can be represented by a linear transformation. Thefore, we perform GE on :math:`A = [M|1]`, with :math:`M` the test matrix and :math:`1` the identiy, and we verify that :math:`M = L^{-1}M'`, where :math:`M', L` are the left and right blocks of :math:`A` after gaussian elimination.
        """
        mat = test_case.matrix
        nrows, ncols = mat.data.shape
        mat_ext = mat.copy()
        mat_ext.concatenate(MatGF2(np.eye(nrows, dtype=np.int_)))
        mat_ext.gauss_elimination(ncols=ncols)
        mat_ge = MatGF2(mat_ext.data[:, :ncols])
        mat_l = MatGF2(mat_ext.data[:, ncols:])

        # Check 1
        p = -1  # pivot
        for i, row in enumerate(mat_ge.data):
            col_idxs = np.flatnonzero(row)  # Column indices with 1s
            if col_idxs.size == 0:
                assert not mat_ge.data[
                    i:, :
                ].any()  # If there aren't any 1s, we verify that the remaining rows are all 0
                break
            j = col_idxs[0]
            assert j > p
            p = j

        # Check 2
        mat_linv = mat_l.right_inverse()
        if mat_linv is not None:
            assert mat_linv @ mat_ge == mat

    @pytest.mark.parametrize("test_case", prepare_test_matrix())
    def test_null_space(self, benchmark: BenchmarkFixture, test_case: LinalgTestCase) -> None:
        mat = test_case.matrix
        kernel_dim = test_case.kernel_dim

        kernel = benchmark(mat.null_space)

        assert kernel_dim == kernel.data.shape[0]
        for v in kernel.data:
            p = mat @ v.transpose()
            assert ~p.data.any()

    @pytest.mark.parametrize("test_case", prepare_test_f2_linear_system())
    def test_solve_f2_linear_system(self, benchmark: BenchmarkFixture, test_case: LSF2TestCase) -> None:
        mat = test_case.mat
        b = test_case.b

        x = benchmark(solve_f2_linear_system, mat, b)

        assert mat @ x == b

    def test_row_reduce(self, fx_rng: Generator) -> None:
        sizes = [(10, 10), (3, 7), (6, 2)]
        ncols = [4, 5, 2]

        for size, ncol in zip(sizes, ncols):
            mat = MatGF2(fx_rng.integers(size=size, low=0, high=2, dtype=np.uint8))
            mat_ref = MatGF2(galois.GF2(mat.data).row_reduce(ncols=ncol))
            mat.row_reduce(ncols=ncol)

            assert mat_ref == mat
