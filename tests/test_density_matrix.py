import random
import unittest
from copy import deepcopy

import numpy as np
import scipy

from graphix import Circuit
from graphix.sim.density_matrix import DensityMatrix, DensityMatrixBackend
from graphix.sim.statevec import CNOT_TENSOR, CZ_TENSOR, SWAP_TENSOR, Statevec, StatevectorBackend


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

        # check with hermitian dm but not unit trace
        with self.assertRaises(ValueError):
            DensityMatrix(data=rand_herm(2 ** np.random.randint(2, 5)))
        # check with non hermitian dm but unit trace
        with self.assertRaises(ValueError):
            l = 2 ** np.random.randint(2, 5)
            tmp = np.random.rand(l, l) + 1j * np.random.rand(l, l)
            DensityMatrix(data=tmp / np.trace(tmp))
        # check with non hermitian dm and not unit trace
        with self.assertRaises(ValueError):
            l = 2 ** np.random.randint(2, 5)  # np.random.randint(2, 20)
            DensityMatrix(data=np.random.rand(l, l) + 1j * np.random.rand(l, l))

        # check not square matrix
        with self.assertRaises(ValueError):
            # l = 2 ** np.random.randint(2, 5) # np.random.randint(2, 20)
            DensityMatrix(data=np.random.rand(3, 2))

        # check higher dimensional matrix
        with self.assertRaises(ValueError):
            DensityMatrix(data=np.random.rand(2, 2, 3))

        # check square and hermitian but with incorrect dimension (non-qubit type)
        with self.assertRaises(ValueError):
            tmp = rand_herm(5)
            DensityMatrix(data=tmp / tmp.trace())

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
            data = rand_herm(2**n)
            data /= np.trace(data)
            dm = DensityMatrix(data=data)
            assert dm.Nqubit == n
            assert dm.rho.shape == (2**n, 2**n)
            assert np.allclose(dm.rho, data)

    def test_evolve_single_fail(self):
        dm = DensityMatrix(nqubit=2)
        # generate random 4 x 4 unitary matrix
        op = rand_unit(4)

        with self.assertRaises(AssertionError):
            dm.evolve_single(op, 2)
        with self.assertRaises(ValueError):
            dm.evolve_single(op, 1)

    def test_evolve_single_success(self):
        # generate random 2 x 2 unitary matrix
        op = rand_unit(2)
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
            data_a = rand_herm(2**n)
            data_a /= np.trace(data_a)
            dm_a = DensityMatrix(data=data_a)
            data_b = rand_herm(2 ** (n + 1))
            data_b /= np.trace(data_b)
            dm_b = DensityMatrix(data=data_b)
            dm_a.tensor(dm_b)
            assert dm_a.Nqubit == 2 * n + 1
            assert dm_a.rho.shape == (2 ** (2 * n + 1), 2 ** (2 * n + 1))
            assert np.allclose(dm_a.rho, np.kron(data_a, data_b))

    def test_cnot_fail(self):
        dm = DensityMatrix(nqubit=2)
        with self.assertRaises(ValueError):
            dm.cnot((1, 1))
        with self.assertRaises(ValueError):
            dm.cnot((-1, 1))
        with self.assertRaises(ValueError):
            dm.cnot((1, -1))
        with self.assertRaises(ValueError):
            dm.cnot((1, 2))
        with self.assertRaises(ValueError):
            dm.cnot((2, 1))

    def test_cnot_success(self):
        dm = DensityMatrix(nqubit=2)
        original_matrix = dm.rho.copy()
        dm.cnot((0, 1))
        expected_matrix = np.array([[1, 1, 1, 1] * 4]).reshape((4, 4)) / 4
        assert np.allclose(dm.rho, expected_matrix)
        dm.cnot((0, 1))
        assert np.allclose(dm.rho, original_matrix)

        # test on 2 qubits only
        psi = np.random.rand(4) + 1j * np.random.rand(4)
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2))
        dm = DensityMatrix(data=np.outer(psi, psi.conj()))
        edge = (0, 1)
        dm.cnot(edge)
        rho = dm.rho.copy()
        psi = psi.reshape((2, 2))
        psi = np.tensordot(CNOT_TENSOR, psi, ((2, 3), edge))
        psi = np.moveaxis(psi, (0, 1), edge)
        expected_matrix = np.outer(psi, psi.conj())
        assert np.allclose(rho, expected_matrix)

        # test on arbitrary number of qubits and random pair
        n = np.random.randint(2, 4)
        psi = np.random.rand(2**n) + 1j * np.random.rand(2**n)
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2))
        dm = DensityMatrix(data=np.outer(psi, psi.conj()))

        edge = tuple(random.sample(range(n), 2))
        dm.cnot(edge)
        rho = dm.rho.copy()
        psi = psi.reshape((2,) * n)
        psi = np.tensordot(CNOT_TENSOR, psi, ((2, 3), edge))
        psi = np.moveaxis(psi, (0, 1), edge)
        expected_matrix = np.outer(psi, psi.conj())
        np.testing.assert_allclose(rho, expected_matrix)

    def test_swap_fail(self):
        dm = DensityMatrix(nqubit=2)
        with self.assertRaises(ValueError):
            dm.swap((1, 1))
        with self.assertRaises(ValueError):
            dm.swap((-1, 1))
        with self.assertRaises(ValueError):
            dm.swap((1, -1))
        with self.assertRaises(ValueError):
            dm.swap((1, 2))
        with self.assertRaises(ValueError):
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
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2))
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
        with self.assertRaises(ValueError):
            dm.entangle((1, 1))
        with self.assertRaises(ValueError):
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
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2))
        dm = DensityMatrix(data=np.outer(psi, psi.conj()))
        edge = (0, 1)
        dm.entangle(edge)
        rho = dm.rho
        psi = psi.reshape((2, 2))
        psi = np.tensordot(CZ_TENSOR, psi, ((2, 3), edge))
        psi = np.moveaxis(psi, (0, 1), edge)
        expected_matrix = np.outer(psi, psi.conj())
        np.testing.assert_allclose(rho, expected_matrix)

    def test_evolve_success(self):

        # single-qubit gate
        # check against evolve_single

        N_qubits = np.random.randint(2, 4)
        N_qubits_op = 1

        # random statevector
        psi = np.random.rand(2**N_qubits) + 1j * np.random.rand(2**N_qubits)
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2))

        # density matrix calculation
        dm = DensityMatrix(data=np.outer(psi, psi.conj()))
        dm_single = deepcopy(dm)

        op = rand_unit(2**N_qubits_op)
        i = np.random.randint(0, N_qubits)

        # need a list format for a single target
        dm.evolve(op, [i])
        dm_single.evolve_single(op, i)

        np.testing.assert_allclose(dm.rho, dm_single.rho)

        # 2-qubit gate

        N_qubits = np.random.randint(2, 4)
        N_qubits_op = 2

        # random unitary
        op = rand_unit(2**N_qubits_op)
        # random pair of indices
        edge = tuple(random.sample(range(N_qubits), 2))

        # random statevector to compare to
        psi = np.random.rand(2**N_qubits) + 1j * np.random.rand(2**N_qubits)
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2))

        # density matrix calculation
        dm = DensityMatrix(data=np.outer(psi, psi.conj()))
        dm.evolve(op, edge)
        rho = dm.rho

        # statevec calculation by hand. 2-qubit op.
        psi = psi.reshape((2,) * N_qubits)
        psi = np.tensordot(op.reshape((2,) * 2 * N_qubits_op), psi, ((2, 3), edge))
        psi = np.moveaxis(psi, (0, 1), edge)
        expected_matrix = np.outer(psi, psi.conj())
        np.testing.assert_allclose(rho, expected_matrix)

        # 3-qubit gate

        N_qubits = np.random.randint(3, 5)
        N_qubits_op = 3

        # random unitary
        op = rand_unit(2**N_qubits_op)
        # 3 random indices
        targets = tuple(random.sample(range(N_qubits), 3))

        # random statevector to compare to
        psi = np.random.rand(2**N_qubits) + 1j * np.random.rand(2**N_qubits)
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2))

        # density matrix calculation
        dm = DensityMatrix(data=np.outer(psi, psi.conj()))
        dm.evolve(op, targets)
        rho = dm.rho

        # statevec calculation by hand. 3-qubit op.
        psi = psi.reshape((2,) * N_qubits)
        psi = np.tensordot(op.reshape((2,) * 2 * N_qubits_op), psi, ((3, 4, 5), targets))
        psi = np.moveaxis(psi, (0, 1, 2), targets)
        expected_matrix = np.outer(psi, psi.conj())
        np.testing.assert_allclose(rho, expected_matrix)

    # TODO by testing evolve, we remove the need for testing indepently CNOT, SWAP and CZ
    def test_evolve_fail(self):

        # test on 3-qubit gate just in case.
        N_qubits = np.random.randint(3, 5)
        N_qubits_op = 3

        # random unitary
        op = rand_unit(2**N_qubits_op)
        # 3 random indices
        targets = tuple(random.sample(range(N_qubits), 3))

        dm = DensityMatrix(nqubit=N_qubits)

        # dimension mismatch
        with self.assertRaises(ValueError):
            dm.evolve(op, (1, 1))
        with self.assertRaises(ValueError):
            dm.evolve(op, (0, 1, 2, 3))
        # incorrect range
        with self.assertRaises(ValueError):
            dm.evolve(op, (-1, 0, 1))
        # repeated index
        with self.assertRaises(ValueError):
            dm.evolve(op, (0, 1, 1))

        # check not square matrix
        with self.assertRaises(ValueError):
            dm.evolve(np.random.rand(2, 3), (0, 1))

        # check higher dimensional matrix
        with self.assertRaises(ValueError):
            dm.evolve(np.random.rand(2, 2, 3), (0, 1))

        # check square but with incorrect dimension (non-qubit type)
        with self.assertRaises(ValueError):
            dm.evolve(np.random.rand(5, 5), (0, 1))

    # TODO the test for normalization is done at initialization with data. Now check that all operations conserve the norm.
    def test_normalize(self):
        #  tmp = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
        data = rand_herm(2 ** np.random.randint(2, 4))
        # data /= data.trace()
        dm = DensityMatrix(data / data.trace())
        dm.normalize()
        assert np.allclose(np.trace(dm.rho), 1)

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

        expected_matrix_1 = np.kron(np.array([[1, 0], [0, 0]]), np.ones((2, 2)) / 2)
        expected_matrix_2 = np.kron(np.array([[0, 0], [0, 1]]), np.array([[0.5, -0.5], [-0.5, 0.5]]))
        assert np.allclose(backend.state.rho, expected_matrix_1) or np.allclose(backend.state.rho, expected_matrix_2)

    def test_correct_byproduct(self):
        np.random.seed(0)

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

    # TODO remove/adapt when Channel class implemented.
    def test_dephase(self):
        def run(p, pattern, max_qubit_num=12):
            backend = DensityMatrixBackend(pattern, max_qubit_num=max_qubit_num)
            for cmd in pattern.seq:
                if cmd[0] == "N":
                    backend.add_nodes([cmd[1]])
                elif cmd[0] == "E":
                    backend.entangle_nodes(cmd[1])
                    backend.dephase(p)
                elif cmd[0] == "M":
                    backend.measure(cmd)
                    backend.dephase(p)
                elif cmd[0] == "X":
                    backend.correct_byproduct(cmd)
                    backend.dephase(p)
                elif cmd[0] == "Z":
                    backend.correct_byproduct(cmd)
                    backend.dephase(p)
                elif cmd[0] == "C":
                    backend.apply_clifford(cmd)
                    backend.dephase(p)
                elif cmd[0] == "T":
                    backend.dephase(p)
                else:
                    raise ValueError("invalid commands")
                if pattern.seq[-1] == cmd:
                    backend.finalize()
            return backend

        # Test for Rx(pi/4)
        circ = Circuit(1)
        circ.rx(0, np.pi / 4)
        pattern = circ.transpile()
        backend1 = run(0, pattern)
        backend2 = run(1, pattern)
        np.testing.assert_allclose(backend1.state.rho, backend2.state.rho)

        # Test for Rz(pi/3)
        circ = Circuit(1)
        circ.rz(0, np.pi / 3)
        pattern = circ.transpile()
        dm_backend = run(1, pattern)
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

        # Test for 3-qubit QFT
        def cp(circuit, theta, control, target):
            """Controlled rotation gate, decomposed"""
            circuit.rz(control, theta / 2)
            circuit.rz(target, theta / 2)
            circuit.cnot(control, target)
            circuit.rz(target, -1 * theta / 2)
            circuit.cnot(control, target)

        def swap(circuit, a, b):
            """swap gate, decomposed"""
            circuit.cnot(a, b)
            circuit.cnot(b, a)
            circuit.cnot(a, b)

        def qft_circ():
            circ = Circuit(3)
            for i in range(3):
                circ.h(i)
            circ.x(1)
            circ.x(2)

            circ.h(2)
            cp(circ, np.pi / 4, 0, 2)
            cp(circ, np.pi / 2, 1, 2)
            circ.h(1)
            cp(circ, np.pi / 2, 0, 1)
            circ.h(0)
            swap(circ, 0, 2)
            return circ

        # no-noise case
        circ = qft_circ()
        pattern = circ.transpile()
        dm_backend = run(0, pattern)
        state = circ.simulate_statevector().flatten()
        np.testing.assert_allclose(dm_backend.state.fidelity(state), 1)

        # noisy case vs exact 3-qubit QFT result
        circ = qft_circ()
        pattern = circ.transpile()
        p = np.random.rand() * 0 + 0.8
        dm_backend = run(p, pattern)
        noisy_state = circ.simulate_statevector().flatten()

        sv = Statevec(nqubit=3)
        omega = np.exp(2j * np.pi / 8)
        qft_matrix = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, omega, omega**2, omega**3, omega**4, omega**5, omega**6, omega**7],
                [1, omega**2, omega**4, omega**6, 1, omega**2, omega**4, omega**6],
                [1, omega**3, omega**6, omega, omega**4, omega**7, omega**2, omega**5],
                [1, omega**4, 1, omega**4, 1, omega**4, 1, omega**4],
                [1, omega**5, omega**2, omega**7, omega**4, omega, omega**6, omega**3],
                [1, omega**6, omega**4, omega**2, 1, omega**6, omega**4, omega**2],
                [1, omega**7, omega**6, omega**5, omega**4, omega**3, omega**2, omega],
            ]
        ) / np.sqrt(8)
        exact_qft_state = qft_matrix @ sv.psi.flatten()
        np.testing.assert_allclose(dm_backend.state.fidelity(noisy_state), dm_backend.state.fidelity(exact_qft_state))


def rand_herm(l: int):
    """
    generate random hermitian matrix of size l*l
    """
    tmp = np.random.rand(l, l) + 1j * np.random.rand(l, l)
    return tmp + tmp.conj().T


def rand_unit(l: int):
    """
    generate random unitary matrix of size l*l from hermitian matrix
    """
    return scipy.linalg.expm(1j * rand_herm(l))


class TestUtilities(unittest.TestCase):

    # not 2**n as for QM but doesn't matter.
    def test_rand_herm(self):
        tmp = rand_herm(np.random.randint(2, 20))
        np.testing.assert_allclose(tmp, tmp.conj().T)

    def test_rand_unit(self):
        d = np.random.randint(2, 20)
        tmp = rand_unit(d)

        # check by applying to a random state
        # can compare both vectors directly since no global phase introduced in the computation.
        psi = np.random.rand(d) + 1j * np.random.rand(d)
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2))
        np.testing.assert_allclose(tmp @ tmp.conj().T @ psi, psi)
        np.testing.assert_allclose(tmp.conj().T @ tmp @ psi, psi)

        # direct assert equal identity doesn't seem to work. Precision issues?
        # np.testing.assert_allclose(tmp @ tmp.conj().T, np.eye(d))
        # np.testing.assert_allclose(tmp.conj().T @ tmp, np.eye(d))


if __name__ == "__main__":
    np.random.seed(2)
    unittest.main()
