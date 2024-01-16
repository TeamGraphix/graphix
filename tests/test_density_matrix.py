import random
import unittest
from copy import deepcopy

import numpy as np

import tests.random_objects as randobj
from graphix import Circuit
from graphix.channels import KrausChannel, dephasing_channel, depolarising_channel
from graphix.ops import Ops
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
            DensityMatrix(data=randobj.rand_herm(2 ** np.random.randint(2, 5)))

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
            # not really a dm since not PSD but ok.
            data = randobj.rand_herm(5)
            data /= np.trace(data)
            DensityMatrix(data=data)

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
        # don't use rand_dm here since want to check
        for n in range(3):
            data = randobj.rand_herm(2**n)

            data /= np.trace(data)
            dm = DensityMatrix(data=data)
            assert dm.Nqubit == n
            assert dm.rho.shape == (2**n, 2**n)
            assert np.allclose(dm.rho, data)

    def test_evolve_single_fail(self):
        dm = DensityMatrix(nqubit=2)
        # generate random 4 x 4 unitary matrix
        op = randobj.rand_unit(4)

        with self.assertRaises(AssertionError):
            dm.evolve_single(op, 2)
        with self.assertRaises(ValueError):
            dm.evolve_single(op, 1)

    def test_evolve_single_success(self):
        # generate random 2 x 2 unitary matrix
        op = randobj.rand_unit(2)

        n = 10
        for i in range(n):
            sv = Statevec(nqubit=n)
            sv.evolve_single(op, i)
            expected_density_matrix = np.outer(sv.psi, sv.psi.conj())
            dm = DensityMatrix(nqubit=n)
            dm.evolve_single(op, i)
            assert np.allclose(dm.rho, expected_density_matrix)

    def test_expectation_single_fail(self):
        nqb = 3
        dm = DensityMatrix(nqubit=nqb)

        # wrong dimensions
        # generate random 4 x 4 unitary matrix
        op = randobj.rand_unit(4)

        with self.assertRaises(ValueError):
            dm.expectation_single(op, 2)
        with self.assertRaises(ValueError):
            dm.expectation_single(op, 1)

        # wrong qubit indices
        op = randobj.rand_unit(2)
        with self.assertRaises(ValueError):
            dm.expectation_single(op, -3)
        with self.assertRaises(ValueError):
            dm.expectation_single(op, nqb + 3)

    def test_expectation_single_success(self):

        """compare to pure state case
        hence only pure states
        but by linearity ok"""

        nqb = np.random.randint(1, 4)
        # NOTE a statevector object so can't use its methods
        target_qubit = np.random.randint(0, nqb)

        psi = np.random.rand(2**nqb) + 1j * np.random.rand(2**nqb)
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2))
        dm = DensityMatrix(data=np.outer(psi, psi.conj()))

        op = randobj.rand_unit(2)

        dm.expectation_single(op, target_qubit)

        # by hand: copy paste from SV backend

        psi1 = np.tensordot(op, psi.reshape((2,) * nqb), (1, target_qubit))
        psi1 = np.moveaxis(psi1, 0, target_qubit)
        psi1 = psi1.reshape(2**nqb)

        # watch out ordering. Expval unitary is cpx so psi1 on the right to match DM.
        np.testing.assert_allclose(np.dot(psi.conjugate(), psi1), dm.expectation_single(op, target_qubit))

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

            data_a = randobj.rand_dm(2**n, dm_dtype=False)
            dm_a = DensityMatrix(data=data_a)

            data_b = randobj.rand_dm(2 ** (n + 1), dm_dtype=False)
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

        # for test only. Sample distinct pairs (== without replacement).
        # https://docs.python.org/2/library/random.html#random.sample

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

        op = randobj.rand_unit(2**N_qubits_op)
        i = np.random.randint(0, N_qubits)

        # need a list format for a single target
        dm.evolve(op, [i])
        dm_single.evolve_single(op, i)

        np.testing.assert_allclose(dm.rho, dm_single.rho)

        # 2-qubit gate

        N_qubits = np.random.randint(2, 4)
        N_qubits_op = 2

        # random unitary
        op = randobj.rand_unit(2**N_qubits_op)

        # random pair of indices
        edge = tuple(random.sample(range(N_qubits), 2))

        # random statevector to compare to
        psi = np.random.rand(2**N_qubits) + 1j * np.random.rand(2**N_qubits)
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2))

        # density matrix calculation
        dm = DensityMatrix(data=np.outer(psi, psi.conj()))
        dm.evolve(op, edge)
        rho = dm.rho

        psi = psi.reshape((2,) * N_qubits)
        psi = np.tensordot(op.reshape((2,) * 2 * N_qubits_op), psi, ((2, 3), edge))
        psi = np.moveaxis(psi, (0, 1), edge)
        expected_matrix = np.outer(psi, psi.conj())
        np.testing.assert_allclose(rho, expected_matrix)

        # 3-qubit gate
        N_qubits = np.random.randint(3, 5)
        N_qubits_op = 3

        # random unitary
        op = randobj.rand_unit(2**N_qubits_op)

        # 3 random indices
        targets = tuple(random.sample(range(N_qubits), 3))

        # random statevector to compare to
        psi = np.random.rand(2**N_qubits) + 1j * np.random.rand(2**N_qubits)
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2))

        # density matrix calculation
        dm = DensityMatrix(data=np.outer(psi, psi.conj()))
        dm.evolve(op, targets)
        rho = dm.rho

        psi = psi.reshape((2,) * N_qubits)
        psi = np.tensordot(op.reshape((2,) * 2 * N_qubits_op), psi, ((3, 4, 5), targets))
        psi = np.moveaxis(psi, (0, 1, 2), targets)
        expected_matrix = np.outer(psi, psi.conj())
        np.testing.assert_allclose(rho, expected_matrix)

    def test_evolve_fail(self):

        # test on 3-qubit gate just in case.
        N_qubits = np.random.randint(3, 5)
        N_qubits_op = 3

        # random unitary
        op = randobj.rand_unit(2**N_qubits_op)
        # 3 random indices
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

        data = randobj.rand_herm(2 ** np.random.randint(2, 4))

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

    def test_apply_dephasing_channel(self):

        # check on single qubit first
        # # create random density matrix
        # data = randobj.rand_herm(2 ** np.random.randint(2, 4))
        data = randobj.rand_herm(2)
        data /= np.trace(data)
        dm = DensityMatrix(data=data)

        # copy of initial dm
        rho_test = deepcopy(dm.rho)

        # create dephasing channel
        prob = np.random.rand()
        dephase_channel = dephasing_channel(prob)

        # useless since checked in apply_channel method.
        assert isinstance(dephase_channel, KrausChannel)
        # useless since checked in the constructor.
        assert dephase_channel.is_normalized()

        # apply channel. list with single element needed.
        # if Channel.nqubit == 1 use list with single element.
        dm.apply_channel(dephase_channel, [0])
        id = np.array([[1.0, 0.0], [0.0, 1.0]])

        # compare
        expected_dm = (
            np.sqrt(1 - prob) ** 2 * id @ rho_test @ id.conj().T
            + np.sqrt(prob) ** 2 * Ops.z @ rho_test @ Ops.z.conj().T
        )

        np.testing.assert_allclose(expected_dm.trace(), 1.0)
        np.testing.assert_allclose(dm.rho, expected_dm)

        N_qubits = np.random.randint(2, 5)

        i = np.random.randint(0, N_qubits)

        # create random density matrix from statevector

        # random statevector to compare to
        psi = np.random.rand(2**N_qubits) + 1j * np.random.rand(2**N_qubits)
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2))

        # build DensityMatrix
        dm = DensityMatrix(data=np.outer(psi, psi.conj()))

        # create dephasing channel
        prob = np.random.rand()
        dephase_channel = dephasing_channel(prob)

        # useless since checked in apply_channel method.
        assert isinstance(dephase_channel, KrausChannel)
        # useless since checked in the constructor.
        assert dephase_channel.is_normalized()

        # apply channel. list with single element needed.
        # if Channel.nqubit == 1 use list with single element.
        dm.apply_channel(dephase_channel, [i])

        # compute on the statevector
        # psi.reshape((2,) * N_qubits)
        # tmp = np.zeros(psi.shape)

        id = np.array([[1.0, 0.0], [0.0, 1.0]])

        # by hand: operator list and gate application
        psi_evolved = np.tensordot(id, psi.reshape((2,) * N_qubits), (1, i))
        psi_evolved = np.moveaxis(psi_evolved, 0, i)

        psi_evolvedb = np.tensordot(Ops.z, psi.reshape((2,) * N_qubits), (1, i))
        psi_evolvedb = np.moveaxis(psi_evolvedb, 0, i)

        # compute final density matrix
        psi_evolved = np.reshape(psi_evolved, (2**N_qubits))
        psi_evolvedb = np.reshape(psi_evolvedb, (2**N_qubits))
        expected_dm = np.sqrt(1 - prob) ** 2 * np.outer(psi_evolved, psi_evolved.conj()) + np.sqrt(
            prob
        ) ** 2 * np.outer(psi_evolvedb, psi_evolvedb.conj())

        # compare
        np.testing.assert_allclose(expected_dm.trace(), 1.0)
        np.testing.assert_allclose(dm.rho, expected_dm)

    def test_apply_depolarising_channel(self):

        # check on single qubit first
        # # create random density matrix
        # data = randobj.rand_herm(2 ** np.random.randint(2, 4))
        data = randobj.rand_herm(2)
        data /= np.trace(data)
        dm = DensityMatrix(data=data)

        # copy of initial dm
        rho_test = deepcopy(dm.rho)

        # create dephasing channel
        prob = np.random.rand()
        depol_channel = depolarising_channel(prob)

        # useless since checked in apply_channel method.
        assert isinstance(depol_channel, KrausChannel)
        # useless since checked in the constructor.
        assert depol_channel.is_normalized()

        # apply channel. list with single element needed.
        # if Channel.nqubit == 1 use list with single element.
        dm.apply_channel(depol_channel, [0])
        id = np.array([[1.0, 0.0], [0.0, 1.0]])

        # compare
        expected_dm = (
            np.sqrt(1 - prob) ** 2 * id @ rho_test @ id.conj().T
            + np.sqrt(prob / 3.0) ** 2 * Ops.x @ rho_test @ Ops.x.conj().T
            + np.sqrt(prob / 3.0) ** 2 * Ops.y @ rho_test @ Ops.y.conj().T
            + np.sqrt(prob / 3.0) ** 2 * Ops.z @ rho_test @ Ops.z.conj().T
        )

        np.testing.assert_allclose(expected_dm.trace(), 1.0)
        np.testing.assert_allclose(dm.rho, expected_dm)

        # chek against statevector backend by hand for now.
        # create random density matrix

        N_qubits = np.random.randint(2, 5)

        # target qubit index
        i = np.random.randint(0, N_qubits)

        # create random density matrix from statevector

        # random statevector to compare to
        psi = np.random.rand(2**N_qubits) + 1j * np.random.rand(2**N_qubits)
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2))

        # build DensityMatrix
        dm = DensityMatrix(data=np.outer(psi, psi.conj()))

        # create dephasing channel
        prob = np.random.rand()
        depol_channel = depolarising_channel(prob)

        # useless since checked in apply_channel method.
        assert isinstance(depol_channel, KrausChannel)
        # useless since checked in the constructor.
        assert depol_channel.is_normalized()

        # apply channel. list with single element needed.
        # if Channel.nqubit == 1 use list with single element.
        dm.apply_channel(depol_channel, [i])

        # compute on the statevector
        # psi.reshape((2,) * N_qubits)
        # tmp = np.zeros(psi.shape)

        id = np.array([[1.0, 0.0], [0.0, 1.0]])

        # by hand: operator list and gate application
        psi_evolved = np.tensordot(id, psi.reshape((2,) * N_qubits), (1, i))
        psi_evolved = np.moveaxis(psi_evolved, 0, i)

        psi_evolvedb = np.tensordot(Ops.x, psi.reshape((2,) * N_qubits), (1, i))
        psi_evolvedb = np.moveaxis(psi_evolvedb, 0, i)

        psi_evolvedc = np.tensordot(Ops.y, psi.reshape((2,) * N_qubits), (1, i))
        psi_evolvedc = np.moveaxis(psi_evolvedc, 0, i)

        psi_evolvedd = np.tensordot(Ops.z, psi.reshape((2,) * N_qubits), (1, i))
        psi_evolvedd = np.moveaxis(psi_evolvedd, 0, i)

        # compute final density matrix
        psi_evolved = np.reshape(psi_evolved, (2**N_qubits))
        psi_evolvedb = np.reshape(psi_evolvedb, (2**N_qubits))
        psi_evolvedc = np.reshape(psi_evolvedc, (2**N_qubits))
        psi_evolvedd = np.reshape(psi_evolvedd, (2**N_qubits))

        expected_dm = (
            np.sqrt(1 - prob) ** 2 * np.outer(psi_evolved, psi_evolved.conj())
            + np.sqrt(prob / 3.0) ** 2 * np.outer(psi_evolvedb, psi_evolvedb.conj())
            + np.sqrt(prob / 3.0) ** 2 * np.outer(psi_evolvedc, psi_evolvedc.conj())
            + np.sqrt(prob / 3.0) ** 2 * np.outer(psi_evolvedd, psi_evolvedd.conj())
        )

        # compare
        np.testing.assert_allclose(expected_dm.trace(), 1.0)
        np.testing.assert_allclose(dm.rho, expected_dm)

    def test_apply_random_channel_one_qubit(self):
        """
        test random 1-qubit channel.
        Especially checks for complex parameters.
        """

        # check against statevector backend by hand for now.
        # create random density matrix

        N_qubits = np.random.randint(2, 5)
        # id = np.array([[1.0, 0.0], [0.0, 1.0]])

        # target qubit index
        i = np.random.randint(0, N_qubits)

        # create random density matrix from statevector

        # random statevector to compare to
        psi = np.random.rand(2**N_qubits) + 1j * np.random.rand(2**N_qubits)
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2))

        # build DensityMatrix
        dm = DensityMatrix(data=np.outer(psi, psi.conj()))

        # create random channel
        # random_channel utility already checked for type and such
        # here dim = 2 (single qubit) and rank is between 1 and 4
        nqb = 1
        dim = 2**nqb
        rk = np.random.randint(1, dim**2 + 1)
        channel = randobj.rand_channel_kraus(dim=dim, rank=rk)

        # apply channel. list with single element needed.
        # if Channel.nqubit == 1 use list with single element.
        dm.apply_channel(channel, [i])

        # compute on the statevector
        # psi.reshape((2,) * N_qubits)
        # tmp = np.zeros(psi.shape)

        # initialize. NOT a DM object, just a matrix.
        expected_dm = np.zeros((2**N_qubits, 2**N_qubits), dtype=np.complex128)

        for elem in channel.kraus_ops:  # kraus_ops is a list of dicts
            psi_evolved = np.tensordot(elem["operator"], psi.reshape((2,) * N_qubits), (1, i))
            psi_evolved = np.moveaxis(psi_evolved, 0, i)
            expected_dm += elem["coef"] * np.conj(elem["coef"]) * np.outer(psi_evolved, np.conj(psi_evolved))

        # compare
        np.testing.assert_allclose(expected_dm.trace(), 1.0)
        np.testing.assert_allclose(dm.rho, expected_dm)

    def test_apply_random_channel_two_qubits(self):
        """
        test random 2-qubit channel on a rank 1 dm (pure state). Generalizes by linearity.
        Especially checks for complex parameters.
        """

        N_qubits = np.random.randint(2, 5)

        # target qubits indices
        qubits = tuple(random.sample(range(N_qubits), 2))

        # create random density matrix from statevector
        # random statevector to compare to
        psi = np.random.rand(2**N_qubits) + 1j * np.random.rand(2**N_qubits)
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2))
        # build DensityMatrix
        dm = DensityMatrix(data=np.outer(psi, psi.conj()))

        # create random channel
        # for 2 qubits, rank between 1 and 16
        # number of qubits it acts on
        nqb = 2
        dim = 2**nqb
        rk = np.random.randint(1, dim**2 + 1)
        channel = randobj.rand_channel_kraus(dim=dim, rank=rk)

        dm.apply_channel(channel, qubits)

        # initialize. NOT a DM object, just a matrix.
        expected_dm = np.zeros((2**N_qubits, 2**N_qubits), dtype=np.complex128)
        # reshape statevec since not in tensor format
        for elem in channel.kraus_ops:  # kraus_ops is a list of dicts
            psi_evolved = np.tensordot(
                elem["operator"].reshape((2,) * 2 * nqb), psi.reshape((2,) * N_qubits), ((2, 3), qubits)
            )
            psi_evolved = np.moveaxis(psi_evolved, (0, 1), qubits)
            expected_dm += elem["coef"] * np.conj(elem["coef"]) * np.outer(psi_evolved, np.conj(psi_evolved))

        np.testing.assert_allclose(expected_dm.trace(), 1.0)
        np.testing.assert_allclose(dm.rho, expected_dm)

    def test_apply_channel_fail(self):
        """
        test apply a channel that is not a Channel object
        """
        N_qubits = np.random.randint(2, 5)
        i = np.random.randint(0, N_qubits)

        psi = np.random.rand(2**N_qubits) + 1j * np.random.rand(2**N_qubits)
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2))

        # build DensityMatrix
        dm = DensityMatrix(data=np.outer(psi, psi.conj()))

        with self.assertRaises(TypeError):
            dm.apply_channel("a", [i])


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

    def test_measure_pr_calc(self):

        # circuit there just to provide a measurement command to try out. Weird.
        circ = Circuit(1)
        circ.rx(0, np.pi / 2)
        pattern = circ.transpile()

        backend = DensityMatrixBackend(pattern, pr_calc=True)
        backend.add_nodes([0, 1, 2])
        backend.entangle_nodes((0, 1))
        backend.entangle_nodes((1, 2))
        backend.measure(backend.pattern.seq[-4])

        # 3-qubit linear graph state: |+0+> + |-1->
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


if __name__ == "__main__":
    np.random.seed(32)
    unittest.main()
