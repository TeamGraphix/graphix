import unittest
import numpy as np
import warnings
from numpy.linalg import norm
from graphix import Circuit
from graphix.sim.statevec import Statevec, StatevectorBackend, CNOT_TENSOR, SWAP_TENSOR, CZ_TENSOR
from graphix.sim.density_matrix import DensityMatrix, DensityMatrixBackend


class TestDensityMatrix(unittest.TestCase):
    """Test for DensityMatrix class."""

    def test_init_without_data_fail(self):
        with self.assertRaises(AssertionError):
            DensityMatrix(nqubit=-2)
        with self.assertRaises(TypeError):
            DensityMatrix(nqubit="hello")
        with self.assertRaises(TypeError):
            DensityMatrix(nqubit=[])

    def test_init_with_invalid_data_fail(self):
        with self.assertRaises(TypeError):
            DensityMatrix("hello")
        with self.assertRaises(TypeError):
            DensityMatrix(1)
        # deprecated data shape (these test might be unnecessary)
        with self.assertRaises(ValueError):
            DensityMatrix([1, 2, [3]])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
            with self.assertRaises(AssertionError):
                DensityMatrix(np.array([1, 2, [3]]))

    def test_init_without_data_success(self):
        for n in range(3):
            dm = DensityMatrix(nqubit=n)
            expected_density_matrix = np.outer(np.ones((2,) * n), np.ones((2,) * n)) / 2**n
            assert dm.Nqubit == n
            assert dm.rho.shape == (2**n, 2**n)
            assert np.allclose(dm.rho, expected_density_matrix)

            dm = DensityMatrix(plus_state=False, nqubit=n)
            expected_density_matrix = np.zeros((2**n, 2**n))
            expected_density_matrix[0, 0] = 1
            assert dm.Nqubit == n
            assert dm.rho.shape == (2**n, 2**n)
            assert np.allclose(dm.rho, expected_density_matrix)

    def test_init_with_data_success(self):
        for n in range(3):
            data = np.random.rand(2**n, 2**n) + 1j * np.random.rand(2**n, 2**n)
            data /= norm(data)
            dm = DensityMatrix(data=data)
            assert dm.Nqubit == n
            assert dm.rho.shape == (2**n, 2**n)
            assert np.allclose(dm.rho, data)

    def test_evolve_single_fail(self):
        dm = DensityMatrix(nqubit=2)
        op = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
        with self.assertRaises(AssertionError):
            dm.evolve_single(op, 2)
        with self.assertRaises(ValueError):
            dm.evolve_single(op, 1)

    def test_evolve_single_success(self):
        op = np.random.rand(2, 2) + 1j * np.random.rand(2, 2)
        n = 10
        for i in range(n):
            sv = Statevec(nqubit=n)
            sv.evolve_single(op, i)
            expected_density_matrix = np.outer(sv.psi, sv.psi.conj())
            dm = DensityMatrix(nqubit=n)
            dm.evolve_single(op, i)
            assert np.allclose(dm.rho, expected_density_matrix)

    def test_tensor_fail(self):
        dm = DensityMatrix(nqubit=1)
        with self.assertRaises(TypeError):
            dm.tensor("hello")
        with self.assertRaises(TypeError):
            dm.tensor(1)

    def test_tensor_without_data_success(self):
        for n in range(3):
            dm_a = DensityMatrix(nqubit=n)
            dm_b = DensityMatrix(nqubit=n + 1)
            dm_a.tensor(dm_b)
            assert dm_a.Nqubit == 2 * n + 1
            assert dm_a.rho.shape == (2 ** (2 * n + 1), 2 ** (2 * n + 1))

    def test_tensor_with_data_success(self):
        for n in range(3):
            data_a = np.random.rand(2**n, 2**n) + 1j * np.random.rand(2**n, 2**n)
            data_a /= norm(data_a)
            dm_a = DensityMatrix(data=data_a)
            data_b = np.random.rand(2 ** (n + 1), 2 ** (n + 1)) + 1j * np.random.rand(2 ** (n + 1), 2 ** (n + 1))
            data_b /= norm(data_b)
            dm_b = DensityMatrix(data=data_b)
            dm_a.tensor(dm_b)
            assert dm_a.Nqubit == 2 * n + 1
            assert dm_a.rho.shape == (2 ** (2 * n + 1), 2 ** (2 * n + 1))
            assert np.allclose(dm_a.rho, np.kron(data_a, data_b))

    def test_cnot_fail(self):
        dm = DensityMatrix(nqubit=2)
        with self.assertRaises(AssertionError):
            dm.cnot((1, 1))
        with self.assertRaises(AssertionError):
            dm.cnot((-1, 1))
        with self.assertRaises(AssertionError):
            dm.cnot((1, -1))
        with self.assertRaises(AssertionError):
            dm.cnot((1, 2))
        with self.assertRaises(AssertionError):
            dm.cnot((2, 1))

    def test_cnot_success(self):
        dm = DensityMatrix(nqubit=2)
        original_matrix = dm.rho.copy()
        dm.cnot((0, 1))
        expected_matrix = np.array([[1, 1, 1, 1] * 4]).reshape((4, 4)) / 4
        assert np.allclose(dm.rho, expected_matrix)
        dm.cnot((0, 1))
        assert np.allclose(dm.rho, original_matrix)

        psi = np.random.rand(4) + 1j * np.random.rand(4)
        dm = DensityMatrix(data=np.outer(psi, psi.conj()))
        edge = (0, 1)
        dm.cnot(edge)
        rho = dm.rho.copy()
        psi = psi.reshape((2, 2))
        psi = np.tensordot(CNOT_TENSOR, psi, ((2, 3), edge))
        psi = np.moveaxis(psi, (0, 1), edge)
        expected_matrix = np.outer(psi, psi.conj())
        assert np.allclose(rho, expected_matrix)

    def test_swap_fail(self):
        dm = DensityMatrix(nqubit=2)
        with self.assertRaises(AssertionError):
            dm.swap((1, 1))
        with self.assertRaises(AssertionError):
            dm.swap((-1, 1))
        with self.assertRaises(AssertionError):
            dm.swap((1, -1))
        with self.assertRaises(AssertionError):
            dm.swap((1, 2))
        with self.assertRaises(AssertionError):
            dm.swap((2, 1))

    def test_swap_success(self):
        dm = DensityMatrix(nqubit=2)
        original_matrix = dm.rho.copy()
        dm.swap((0, 1))
        expected_matrix = np.array([[1, 1, 1, 1] * 4]).reshape((4, 4)) / 4
        assert np.allclose(dm.rho, expected_matrix)
        dm.swap((0, 1))
        assert np.allclose(dm.rho, original_matrix)

        psi = np.random.rand(4) + 1j * np.random.rand(4)
        dm = DensityMatrix(data=np.outer(psi, psi.conj()))
        edge = (0, 1)
        dm.swap(edge)
        rho = dm.rho
        psi = psi.reshape((2, 2))
        psi = np.tensordot(SWAP_TENSOR, psi, ((2, 3), edge))
        psi = np.moveaxis(psi, (0, 1), edge)
        expected_matrix = np.outer(psi, psi.conj())
        assert np.allclose(rho, expected_matrix)

    def test_entangle_fail(self):
        dm = DensityMatrix(nqubit=3)
        with self.assertRaises(AssertionError):
            dm.entangle((1, 1))
        with self.assertRaises(AssertionError):
            dm.entangle(((1, 3)))
        with self.assertRaises(ValueError):
            dm.entangle((0, 1, 2))

    def test_entangle_success(self):
        dm = DensityMatrix(nqubit=2)
        original_matrix = dm.rho.copy()
        dm.entangle((0, 1))
        expected_matrix = np.array([[1, 1, 1, -1], [1, 1, 1, -1], [1, 1, 1, -1], [-1, -1, -1, 1]]) / 4
        assert np.allclose(dm.rho, expected_matrix)
        dm.entangle((0, 1))
        assert np.allclose(dm.rho, original_matrix)

        dm = DensityMatrix(nqubit=3)
        dm.entangle((0, 1))
        dm.entangle((2, 1))
        dm1 = dm.rho.copy()

        dm = DensityMatrix(nqubit=3)
        dm.entangle((2, 1))
        dm.entangle((0, 1))
        dm2 = dm.rho
        assert np.allclose(dm1, dm2)

        psi = np.random.rand(4) + 1j * np.random.rand(4)
        dm = DensityMatrix(data=np.outer(psi, psi.conj()))
        edge = (0, 1)
        dm.entangle(edge)
        rho = dm.rho
        psi = psi.reshape((2, 2))
        psi = np.tensordot(CZ_TENSOR, psi, ((2, 3), edge))
        psi = np.moveaxis(psi, (0, 1), edge)
        expected_matrix = np.outer(psi, psi.conj())
        assert np.allclose(rho, expected_matrix)

    def test_normalize(self):
        data = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
        dm = DensityMatrix(data)
        dm.normalize()
        assert np.allclose(norm(dm.rho), 1)

    def test_ptrace_fail(self):
        dm = DensityMatrix(nqubit=0)
        with self.assertRaises(AssertionError):
            dm.ptrace((0,))
        dm = DensityMatrix(nqubit=2)
        with self.assertRaises(AssertionError):
            dm.ptrace((2,))

    def test_ptrace(self):
        psi = np.kron(np.array([1, 0]), np.array([1, 1]) / np.sqrt(2))
        data = np.outer(psi, psi)
        dm = DensityMatrix(data=data)
        dm.ptrace((0,))
        expected_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
        assert np.allclose(dm.rho, expected_matrix)

        dm = DensityMatrix(data=data)
        dm.ptrace((1,))
        expected_matrix = np.array([[1, 0], [0, 0]])
        assert np.allclose(dm.rho, expected_matrix)

        dm = DensityMatrix(data=data)
        dm.ptrace((0, 1))
        expected_matrix = np.array([1])
        assert np.allclose(dm.rho, expected_matrix)

        dm = DensityMatrix(nqubit=4)
        dm.ptrace((0, 1, 2))
        expected_matrix = np.array([[1, 1], [1, 1]]) / 2
        assert np.allclose(dm.rho, expected_matrix)

        dm = DensityMatrix(nqubit=4)
        dm.ptrace((0, 1, 2, 3))
        expected_matrix = np.array([1])
        assert np.allclose(dm.rho, expected_matrix)

        psi = np.kron(np.kron(np.array([1, np.sqrt(2)]) / np.sqrt(3), np.array([1, 0])), np.array([0, 1]))
        data = np.outer(psi, psi)
        dm = DensityMatrix(data=data)
        dm.ptrace((2,))
        expected_matrix = np.array(
            [[1 / 3, 0, np.sqrt(2) / 3, 0], [0, 0, 0, 0], [np.sqrt(2) / 3, 0, 2 / 3, 0], [0, 0, 0, 0]]
        )
        assert np.allclose(dm.rho, expected_matrix)


class DensityMatrixBackendTest(unittest.TestCase):
    """Test for DensityMatrixBackend class."""

    def test_init_fail(self):
        with self.assertRaises(TypeError):
            DensityMatrixBackend()

    def test_init_success(self):
        circ = Circuit(1)
        circ.rx(0, np.pi / 2)
        pattern = circ.transpile()
        backend = DensityMatrixBackend(pattern)
        assert backend.pattern == pattern
        assert backend.results == pattern.results
        assert backend.state is None
        assert backend.node_index == []
        assert backend.Nqubit == 0
        assert backend.max_qubit_num == 12

    def test_add_nodes(self):
        circ = Circuit(1)
        pattern = circ.transpile()
        backend = DensityMatrixBackend(pattern)
        backend.add_nodes([0, 1])
        expected_matrix = np.array([0.25] * 16).reshape(4, 4)
        np.testing.assert_allclose(backend.state.rho, expected_matrix)

    def test_entangle_nodes(self):
        circ = Circuit(1)
        pattern = circ.transpile()
        backend = DensityMatrixBackend(pattern)
        backend.add_nodes([0, 1])
        backend.entangle_nodes((0, 1))
        expected_matrix = np.array([[1, 1, 1, -1], [1, 1, 1, -1], [1, 1, 1, -1], [-1, -1, -1, 1]]) / 4
        np.testing.assert_allclose(backend.state.rho, expected_matrix)

        backend.entangle_nodes((0, 1))
        np.testing.assert_allclose(backend.state.rho, np.array([0.25] * 16).reshape(4, 4))

    def test_measure(self):
        circ = Circuit(1)
        circ.rx(0, np.pi / 2)
        pattern = circ.transpile()

        backend = DensityMatrixBackend(pattern)
        backend.add_nodes([0, 1, 2])
        backend.entangle_nodes((0, 1))
        backend.entangle_nodes((1, 2))
        backend.measure(backend.pattern.seq[-4])

        expected_matrix_1 = np.zeros((4, 4))
        expected_matrix_1[0, 0] = 0.5
        expected_matrix_1[0, 1] = 0.5
        expected_matrix_1[1, 0] = 0.5
        expected_matrix_1[1, 1] = 0.5
        expected_matrix_2 = np.zeros((4, 4))
        expected_matrix_2[2, 2] = 0.5
        expected_matrix_2[2, 3] = -0.5
        expected_matrix_2[3, 2] = -0.5
        expected_matrix_2[3, 3] = 0.5

        assert np.allclose(backend.state.rho, expected_matrix_1) or np.allclose(backend.state.rho, expected_matrix_2)

    def test_correct_byproduct(self):
        circ = Circuit(1)
        circ.rx(0, np.pi / 2)
        pattern = circ.transpile()

        backend = DensityMatrixBackend(pattern)
        backend.add_nodes([0, 1, 2])
        backend.entangle_nodes((0, 1))
        backend.entangle_nodes((1, 2))
        backend.measure(backend.pattern.seq[-4])
        backend.measure(backend.pattern.seq[-3])
        backend.correct_byproduct(backend.pattern.seq[-2])
        backend.correct_byproduct(backend.pattern.seq[-1])
        backend.finalize()
        rho = backend.state.rho

        backend = StatevectorBackend(pattern)
        backend.add_nodes([0, 1, 2])
        backend.entangle_nodes((0, 1))
        backend.entangle_nodes((1, 2))
        backend.measure(backend.pattern.seq[-4])
        backend.measure(backend.pattern.seq[-3])
        backend.correct_byproduct(backend.pattern.seq[-2])
        backend.correct_byproduct(backend.pattern.seq[-1])
        backend.finalize()
        psi = backend.state.psi

        np.testing.assert_allclose(rho, np.outer(psi, psi.conj()))

    def test_dephase(self):
        def simulate(p, pattern):
            backend = DensityMatrixBackend(pattern)
            backend.add_nodes([0])
            backend.dephase(p)
            backend.add_nodes([1])
            backend.dephase(p)
            backend.add_nodes([2])
            backend.dephase(p)
            backend.entangle_nodes((0, 1))
            backend.dephase(p)
            backend.entangle_nodes((1, 2))
            backend.dephase(p)
            backend.measure(pattern.seq[-4])
            backend.dephase(p)
            backend.measure(pattern.seq[-3])
            backend.dephase(p)
            backend.correct_byproduct(pattern.seq[-2])
            backend.dephase(p)
            backend.correct_byproduct(pattern.seq[-1])
            backend.dephase(p)
            backend.finalize()
            return backend

        circ = Circuit(1)
        circ.rx(0, np.pi / 4)
        pattern = circ.transpile()
        backend1 = simulate(0, pattern)
        backend2 = simulate(1, pattern)
        np.testing.assert_allclose(backend1.state.rho, backend2.state.rho)

        circ = Circuit(1)
        circ.rz(0, np.pi / 3)
        pattern = circ.transpile()
        dm_backend = simulate(1, pattern)
        sv_backend = StatevectorBackend(pattern)
        sv_backend.add_nodes([0, 1, 2])
        sv_backend.entangle_nodes((0, 1))
        sv_backend.entangle_nodes((1, 2))
        sv_backend.measure(pattern.seq[-4])
        sv_backend.measure(pattern.seq[-3])
        sv_backend.correct_byproduct(pattern.seq[-2])
        sv_backend.correct_byproduct(pattern.seq[-1])
        sv_backend.finalize()
        np.testing.assert_allclose(dm_backend.state.fidelity(sv_backend.state.psi), 0.25)


unittest.main(argv=[""], verbosity=1, exit=False)
