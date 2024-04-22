import galois
import numpy as np
import pytest

from graphix.linalg import MatGF2


def prepare_test_matrix():
    test_cases = []

    # empty matrix
    test_case = dict()
    test_case["matrix"] = MatGF2(np.array([[]], dtype=int))
    test_case["forward_eliminated"] = np.array([[]], dtype=int)
    test_case["rank"] = 0
    test_case["RHS_input"] = np.array([[]], dtype=int)
    test_case["RHS_forward_elimnated"] = np.array([[]], dtype=int)
    test_case["x"] = [[]]
    test_case["kernel_dim"] = 0
    test_cases.append(test_case)

    # column vector
    test_case = dict()
    test_case["matrix"] = MatGF2(np.array([[1], [1], [1]], dtype=int))
    test_case["forward_eliminated"] = np.array([[1], [0], [0]], dtype=int)
    test_case["rank"] = 1
    test_case["RHS_input"] = np.array([[1], [1], [1]], dtype=int)
    test_case["RHS_forward_elimnated"] = np.array([[1], [0], [0]], dtype=int)
    test_case["x"] = [[1]]
    test_case["kernel_dim"] = 0
    test_cases.append(test_case)

    # row vector
    test_case = dict()
    test_case["matrix"] = MatGF2(np.array([[1, 1, 1]], dtype=int))
    test_case["forward_eliminated"] = np.array([[1, 1, 1]], dtype=int)
    test_case["rank"] = 1
    test_case["RHS_input"] = np.array([[1]], dtype=int)
    test_case["RHS_forward_elimnated"] = np.array([[1]], dtype=int)
    test_case["x"] = None  # TODO: add x
    test_case["kernel_dim"] = 2
    test_cases.append(test_case)

    # diagonal matrix
    test_case = dict()
    test_case["matrix"] = MatGF2(np.diag(np.ones(10)).astype(int))
    test_case["forward_eliminated"] = np.diag(np.ones(10)).astype(int)
    test_case["rank"] = 10
    test_case["RHS_input"] = np.ones(10).reshape(10, 1).astype(int)
    test_case["RHS_forward_elimnated"] = np.ones(10).reshape(10, 1).astype(int)
    test_case["x"] = list(np.ones((10, 1)))
    test_case["kernel_dim"] = 0
    test_cases.append(test_case)

    # full rank dense matrix
    test_case = dict()
    test_case["matrix"] = MatGF2(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=int))
    test_case["forward_eliminated"] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int)
    test_case["rank"] = 3
    test_case["RHS_input"] = np.array([[1], [1], [1]], dtype=int)
    test_case["RHS_forward_elimnated"] = np.array([[1], [1], [0]], dtype=int)
    test_case["x"] = list(np.array([[1], [1], [0]]))  # nan for no solution
    test_case["kernel_dim"] = 0
    test_cases.append(test_case)

    # not full-rank matrix
    test_case = dict()
    test_case["matrix"] = MatGF2(np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=int))
    test_case["forward_eliminated"] = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 0]], dtype=int)
    test_case["rank"] = 2
    test_case["RHS_input"] = np.array([[1, 1], [1, 1], [0, 1]], dtype=int)
    test_case["RHS_forward_elimnated"] = np.array([[1, 1], [1, 1], [0, 1]], dtype=int)
    test_case["x"] = None  # TODO: add x
    test_case["kernel_dim"] = 1
    test_cases.append(test_case)

    # non-square matrix
    test_case = dict()
    test_case["matrix"] = MatGF2(np.array([[1, 0, 1], [0, 1, 0]], dtype=int))
    test_case["forward_eliminated"] = np.array([[1, 0, 1], [0, 1, 0]], dtype=int)
    test_case["rank"] = 2
    test_case["RHS_input"] = np.array([[1], [1]], dtype=int)
    test_case["RHS_forward_elimnated"] = np.array([[1], [1]], dtype=int)
    test_case["x"] = None  # TODO: add x
    test_case["kernel_dim"] = 1
    test_cases.append(test_case)

    # non-square matrix
    test_case = dict()
    test_case["matrix"] = MatGF2(np.array([[1, 0], [0, 1], [1, 0]], dtype=int))
    test_case["forward_eliminated"] = np.array([[1, 0], [0, 1], [0, 0]], dtype=int)
    test_case["rank"] = 2
    test_case["RHS_input"] = np.array([[1], [1], [1]], dtype=int)
    test_case["RHS_forward_elimnated"] = np.array([[1], [1], [0]], dtype=int)
    test_case["x"] = [[1], [1]]
    test_case["kernel_dim"] = 0
    test_cases.append(test_case)

    return test_cases


class TestLinAlg:
    def test_add_row(self):
        test_mat = MatGF2(np.diag(np.ones(2, dtype=int)))
        test_mat.add_row()
        assert test_mat.data.shape == (3, 2)
        assert np.all(test_mat.data == np.array([[1, 0], [0, 1], [0, 0]]))

    def test_add_col(self):
        test_mat = MatGF2(np.diag(np.ones(2, dtype=int)))
        test_mat.add_col()
        assert test_mat.data.shape == (2, 3)
        assert np.all(test_mat.data == galois.GF2(np.array([[1, 0, 0], [0, 1, 0]])))

    def test_remove_row(self):
        test_mat = MatGF2(np.array([[1, 0], [0, 1], [0, 0]], dtype=int))
        test_mat.remove_row(2)
        assert np.all(test_mat.data == np.array([[1, 0], [0, 1]]))

    def test_remove_col(self):
        test_mat = MatGF2(np.array([[1, 0, 0], [0, 1, 0]], dtype=int))
        test_mat.remove_col(2)
        assert np.all(test_mat.data == np.array([[1, 0], [0, 1]]))

    def test_swap_row(self):
        test_mat = MatGF2(np.array([[1, 0], [0, 1], [0, 0]], dtype=int))
        test_mat.swap_row(1, 2)
        assert np.all(test_mat.data == np.array([[1, 0], [0, 0], [0, 1]]))

    def test_swap_col(self):
        test_mat = MatGF2(np.array([[1, 0, 0], [0, 1, 0]], dtype=int))
        test_mat.swap_col(1, 2)
        assert np.all(test_mat.data == np.array([[1, 0, 0], [0, 0, 1]]))

    def test_is_canonical_form(self):
        test_mat = MatGF2(np.array([[1, 0], [0, 1]], dtype=int))
        assert test_mat.is_canonical_form()

        test_mat = MatGF2(np.array([[1, 0], [0, 1], [0, 0]], dtype=int))
        assert test_mat.is_canonical_form()

        test_mat = MatGF2(np.array([[1, 0, 1], [0, 1, 0], [0, 0, 0]], dtype=int))
        assert test_mat.is_canonical_form()

        test_mat = MatGF2(np.array([[1, 1, 0], [0, 0, 1], [0, 1, 0]], dtype=int))
        assert not test_mat.is_canonical_form()

    @pytest.mark.parametrize("test_case", prepare_test_matrix())
    def test_forward_eliminate(self, test_case):
        mat = test_case["matrix"]
        answer = test_case["forward_eliminated"]
        RHS_input = test_case["RHS_input"]
        RHS_forward_elimnated = test_case["RHS_forward_elimnated"]
        mat_elimnated, RHS, _, _ = mat.forward_eliminate(RHS_input)
        assert np.all(mat_elimnated.data == answer)
        assert np.all(RHS.data == RHS_forward_elimnated)

    @pytest.mark.parametrize("test_case", prepare_test_matrix())
    def test_get_rank(self, test_case):
        mat = test_case["matrix"]
        rank = test_case["rank"]
        assert mat.get_rank() == rank

    @pytest.mark.parametrize("test_case", prepare_test_matrix())
    def test_backward_substitute(self, test_case):
        mat = test_case["matrix"]
        RHS_input = test_case["RHS_input"]
        x = test_case["x"]
        kernel_dim = test_case["kernel_dim"]
        mat_eliminated, RHS_eliminated, _, _ = mat.forward_eliminate(RHS_input)
        x, kernel = mat_eliminated.backward_substitute(RHS_eliminated)
        if x is not None:
            assert np.all(x == x)
        assert len(kernel) == kernel_dim
