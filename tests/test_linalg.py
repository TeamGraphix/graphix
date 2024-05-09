from __future__ import annotations

from typing import NamedTuple

import galois
import numpy as np
import numpy.typing as npt
import pytest

from graphix.linalg import MatGF2


class LinalgTestCase(NamedTuple):
    matrix: MatGF2
    forward_eliminated: npt.NDArray[np.int_]
    rank: int
    rhs_input: npt.NDArray[np.int_]
    rhs_forward_eliminated: npt.NDArray[np.int_]
    x: list[npt.NDArray[np.int_]] | None
    kernel_dim: int


def prepare_test_matrix() -> list[LinalgTestCase]:
    return [
        # empty matrix
        LinalgTestCase(
            MatGF2(np.array([[]], dtype=np.int_)),
            np.array([[]], dtype=np.int_),
            0,
            np.array([[]], dtype=np.int_),
            np.array([[]], dtype=np.int_),
            [np.array([], dtype=np.int_)],
            0,
        ),
        # column vector
        LinalgTestCase(
            MatGF2(np.array([[1], [1], [1]], dtype=np.int_)),
            np.array([[1], [0], [0]], dtype=np.int_),
            1,
            np.array([[1], [1], [1]], dtype=np.int_),
            np.array([[1], [0], [0]], dtype=np.int_),
            [np.array([1])],
            0,
        ),
        # row vector
        LinalgTestCase(
            MatGF2(np.array([[1, 1, 1]], dtype=np.int_)),
            np.array([[1, 1, 1]], dtype=np.int_),
            1,
            np.array([[1]], dtype=np.int_),
            np.array([[1]], dtype=np.int_),
            None,  # TODO: add x
            2,
        ),
        # diagonal matrix
        LinalgTestCase(
            MatGF2(np.diag(np.ones(10)).astype(int)),
            np.diag(np.ones(10)).astype(int),
            10,
            np.ones(10).reshape(10, 1).astype(int),
            np.ones(10).reshape(10, 1).astype(int),
            list(np.ones((10, 1))),
            0,
        ),
        # full rank dense matrix
        LinalgTestCase(
            MatGF2(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.int_)),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int_),
            3,
            np.array([[1], [1], [1]], dtype=np.int_),
            np.array([[1], [1], [0]], dtype=np.int_),
            list(np.array([[1], [1], [0]])),  # nan for no solution
            0,
        ),
        # not full-rank matrix
        LinalgTestCase(
            MatGF2(np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=np.int_)),
            np.array([[1, 0, 1], [0, 1, 0], [0, 0, 0]], dtype=np.int_),
            2,
            np.array([[1, 1], [1, 1], [0, 1]], dtype=np.int_),
            np.array([[1, 1], [1, 1], [0, 1]], dtype=np.int_),
            None,  # TODO: add x
            1,
        ),
        # non-square matrix
        LinalgTestCase(
            MatGF2(np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int_)),
            np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int_),
            2,
            np.array([[1], [1]], dtype=np.int_),
            np.array([[1], [1]], dtype=np.int_),
            None,  # TODO: add x
            1,
        ),
        # non-square matrix
        LinalgTestCase(
            MatGF2(np.array([[1, 0], [0, 1], [1, 0]], dtype=np.int_)),
            np.array([[1, 0], [0, 1], [0, 0]], dtype=np.int_),
            2,
            np.array([[1], [1], [1]], dtype=np.int_),
            np.array([[1], [1], [0]], dtype=np.int_),
            [np.array([1], dtype=np.int_), np.array([1], dtype=np.int_)],
            0,
        ),
    ]


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

    def test_is_canonical_form(self) -> None:
        test_mat = MatGF2(np.array([[1, 0], [0, 1]], dtype=np.int_))
        assert test_mat.is_canonical_form()

        test_mat = MatGF2(np.array([[1, 0], [0, 1], [0, 0]], dtype=np.int_))
        assert test_mat.is_canonical_form()

        test_mat = MatGF2(np.array([[1, 0, 1], [0, 1, 0], [0, 0, 0]], dtype=np.int_))
        assert test_mat.is_canonical_form()

        test_mat = MatGF2(np.array([[1, 1, 0], [0, 0, 1], [0, 1, 0]], dtype=np.int_))
        assert not test_mat.is_canonical_form()

    @pytest.mark.parametrize("test_case", prepare_test_matrix())
    def test_forward_eliminate(self, test_case: LinalgTestCase) -> None:
        mat = test_case.matrix
        answer = test_case.forward_eliminated
        rhs_input = test_case.rhs_input
        rhs_forward_elimnated = test_case.rhs_forward_eliminated
        mat_elimnated, rhs, _, _ = mat.forward_eliminate(rhs_input)
        assert np.all(mat_elimnated.data == answer)
        assert np.all(rhs.data == rhs_forward_elimnated)

    @pytest.mark.parametrize("test_case", prepare_test_matrix())
    def test_get_rank(self, test_case: LinalgTestCase) -> None:
        mat = test_case.matrix
        rank = test_case.rank
        assert mat.get_rank() == rank

    @pytest.mark.parametrize("test_case", prepare_test_matrix())
    def test_backward_substitute(self, test_case: LinalgTestCase) -> None:
        mat = test_case.matrix
        rhs_input = test_case.rhs_input
        x = test_case.x
        kernel_dim = test_case.kernel_dim
        mat_eliminated, rhs_eliminated, _, _ = mat.forward_eliminate(rhs_input)
        x, kernel = mat_eliminated.backward_substitute(rhs_eliminated)
        if x is not None:
            assert np.all(x == x)
        assert len(kernel) == kernel_dim
