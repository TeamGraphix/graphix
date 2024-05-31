from __future__ import annotations

import functools
import random
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest

import graphix.pauli
import graphix.random_objects as randobj
import graphix.states
from graphix import Circuit
from graphix.channels import KrausChannel, dephasing_channel, depolarising_channel
from graphix.ops import Ops
from graphix.sim.density_matrix import DensityMatrix, DensityMatrixBackend
from graphix.sim.statevec import CNOT_TENSOR, CZ_TENSOR, SWAP_TENSOR, Statevec, StatevectorBackend

if TYPE_CHECKING:
    from numpy.random import Generator


def _randstate_raw(nqubits: int, rng: Generator) -> npt.NDArray[np.complex128]:
    size = (2**nqubits,)
    return rng.uniform(size=size) + rng.uniform(size=size) * 1j


def _randdm_raw(nqubits: int, rng: Generator) -> npt.NDArray[np.complex128]:
    size = (2**nqubits, 2**nqubits)
    return rng.uniform(size=size) + rng.uniform(size=size) * 1j


class TestDensityMatrix:
    """Test for DensityMatrix class."""

    def test_init_without_data_fail(self) -> None:
        with pytest.raises(AssertionError):
            DensityMatrix(nqubit=-2)
        with pytest.raises(AssertionError):
            DensityMatrix(nqubit="hello")
        with pytest.raises(AssertionError):
            DensityMatrix(nqubit=[])

    def test_init_with_invalid_data_fail(self, fx_rng: Generator) -> None:
        with pytest.raises(TypeError):
            DensityMatrix("hello")
        with pytest.raises(TypeError):
            DensityMatrix(1)
        with pytest.raises(TypeError):
            DensityMatrix([1, 2, [3]])

        # check with hermitian dm but not unit trace
        with pytest.raises(ValueError):
            DensityMatrix(data=randobj.rand_herm(2 ** fx_rng.integers(2, 5)))
        # check with non hermitian dm but unit trace
        tmp = _randdm_raw(fx_rng.integers(2, 5), fx_rng)
        with pytest.raises(ValueError):
            DensityMatrix(data=tmp / np.trace(tmp))
        # check with non hermitian dm and not unit trace
        with pytest.raises(ValueError):
            DensityMatrix(data=_randdm_raw(fx_rng.integers(2, 5), fx_rng))
        # check not square matrix
        with pytest.raises(ValueError):
            # l = 2 ** fx_rng.integers(2, 5) # fx_rng.integers(2, 20)
            DensityMatrix(data=fx_rng.uniform(size=(3, 2)))
        # check higher dimensional matrix
        with pytest.raises(TypeError):
            DensityMatrix(data=fx_rng.uniform(size=(2, 2, 3)))
        # check square and hermitian but with incorrect dimension (non-qubit type)
        data = randobj.rand_herm(5)
        data /= np.trace(data)
        with pytest.raises(ValueError):
            # not really a dm since not PSD but ok.
            DensityMatrix(data=data)

    @pytest.mark.parametrize("n", range(3))
    def test_init_without_data_success(self, n: int) -> None:
        dm = DensityMatrix(nqubit=n)
        expected_density_matrix = np.outer(np.ones((2,) * n), np.ones((2,) * n)) / 2**n
        assert dm.Nqubit == n
        assert dm.rho.shape == (2**n, 2**n)
        assert np.allclose(dm.rho, expected_density_matrix)

        dm = DensityMatrix(data=graphix.states.BasicStates.ZERO, nqubit=n)
        expected_density_matrix = np.zeros((2**n, 2**n))
        expected_density_matrix[0, 0] = 1
        assert dm.Nqubit == n
        assert dm.rho.shape == (2**n, 2**n)
        assert np.allclose(dm.rho, expected_density_matrix)

    # TODO: Use pytest.mark.parametrize after refactoring randobj.rand_herm
    def test_init_with_data_success(self) -> None:
        # don't use rand_dm here since want to check
        for n in range(3):
            _data = randobj.rand_herm(2**n)

    def test_init_with_state_sucess(self, fx_rng: Generator) -> None:
        # both "numerical" statevec and Statevec object
        # relies on Statevec constructor validation

        nqb = fx_rng.integers(2, 5)
        print(f"nqb is {nqb}")
        rand_angles = fx_rng.random(nqb) * 2 * np.pi
        rand_planes = fx_rng.choice([i for i in graphix.pauli.Plane], nqb)
        states = [graphix.states.PlanarState(plane=i, angle=j) for i, j in zip(rand_planes, rand_angles)]
        vec = Statevec(data=states)
        # flattens input!
        expected_dm = np.outer(vec.psi, vec.psi.conj())

        # input with a State object
        dm = DensityMatrix(data=states)
        assert dm.dims() == (2**nqb, 2**nqb)
        assert np.allclose(dm.rho, expected_dm)

    def test_init_with_state_fail(self, fx_rng: Generator) -> None:
        nqb = 2
        rand_angles = fx_rng.random(nqb) * 2 * np.pi
        rand_planes = fx_rng.choice(np.array([i for i in graphix.pauli.Plane]), nqb)
        states = [graphix.states.PlanarState(plane=i, angle=j) for i, j in zip(rand_planes, rand_angles)]

        with pytest.raises(ValueError):
            _dm = DensityMatrix(nqubit=1, data=states)

        with pytest.raises(ValueError):
            _dm = DensityMatrix(nqubit=3, data=states)

    def test_init_with_statevec_sucess(self, fx_rng: Generator) -> None:
        # both "numerical" statevec and Statevec object
        # relies on Statevec constructor validation

        nqb = fx_rng.integers(2, 5)
        rand_angles = fx_rng.random(nqb) * 2 * np.pi
        rand_planes = fx_rng.choice(np.array([i for i in graphix.pauli.Plane]), nqb)
        states = [graphix.states.PlanarState(plane=i, angle=j) for i, j in zip(rand_planes, rand_angles)]
        vec = Statevec(data=states)
        # flattens input!
        expected_dm = np.outer(vec.psi, vec.psi.conj())

        # input with a Statevec object
        dm = DensityMatrix(data=vec)
        assert dm.dims() == (2**nqb, 2**nqb)
        assert np.allclose(dm.rho, expected_dm)

        sv_list = [state.get_statevector() for state in states]
        sv = functools.reduce(np.kron, sv_list)

        # input with a statevector DATA (not Statevec object)
        dm2 = DensityMatrix(data=sv)

        print("dims", dm.dims())
        assert dm2.dims() == (2**nqb, 2**nqb)
        assert np.allclose(dm2.rho, dm.rho)
        assert np.allclose(dm2.rho, expected_dm)

    def test_init_with_densitymatrix_sucess(self, fx_rng: Generator) -> None:
        # both "numerical" densitymatrix and DensityMatrix object

        nqb = fx_rng.integers(2, 5)
        rand_angles = fx_rng.random(nqb) * 2 * np.pi
        rand_planes = fx_rng.choice(np.array([i for i in graphix.pauli.Plane]), nqb)
        print("planes", rand_planes)
        states = [graphix.states.PlanarState(plane=i, angle=j) for i, j in zip(rand_planes, rand_angles)]
        vec = Statevec(data=states)
        expected_dm = np.outer(vec.psi, vec.psi.conj())

        # input with a huge density matrix
        dm_list = [state.get_densitymatrix() for state in states]
        num_dm = functools.reduce(np.kron, dm_list)

        dm = DensityMatrix(data=num_dm)

        assert dm.dims() == (2**nqb, 2**nqb)
        assert np.allclose(dm.rho, expected_dm)

        # check copying
        dm2 = DensityMatrix(dm)
        assert dm2.dims() == (2**nqb, 2**nqb)
        assert np.allclose(dm2.rho, expected_dm)
        assert np.allclose(dm2.rho, dm.rho)

    def test_evolve_single_fail(self) -> None:
        dm = DensityMatrix(nqubit=2)
        # generate random 4 x 4 unitary matrix
        op = randobj.rand_unit(4)

        with pytest.raises(AssertionError):
            dm.evolve_single(op, 2)
        with pytest.raises(ValueError):
            dm.evolve_single(op, 1)

    def test_evolve_single_success(self) -> None:
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

    def test_expectation_single_fail(self) -> None:
        nqb = 3
        dm = DensityMatrix(nqubit=nqb)

        # wrong dimensions
        # generate random 4 x 4 unitary matrix
        op = randobj.rand_unit(4)

        with pytest.raises(ValueError):
            dm.expectation_single(op, 2)
        with pytest.raises(ValueError):
            dm.expectation_single(op, 1)

        # wrong qubit indices
        op = randobj.rand_unit(2)
        with pytest.raises(ValueError):
            dm.expectation_single(op, -3)
        with pytest.raises(ValueError):
            dm.expectation_single(op, nqb + 3)

    def test_expectation_single_success(self, fx_rng: Generator) -> None:
        """compare to pure state case
        hence only pure states
        but by linearity ok"""

        nqb = fx_rng.integers(1, 4)
        # NOTE a statevector object so can't use its methods
        target_qubit = fx_rng.integers(0, nqb)

        psi = _randstate_raw(nqb, fx_rng)
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2))
        dm = DensityMatrix(data=np.outer(psi, psi.conj()))

        op = randobj.rand_unit(2)

        dm.expectation_single(op, target_qubit)

        # by hand: copy paste from SV backend

        psi1 = np.tensordot(op, psi.reshape((2,) * nqb), (1, target_qubit))
        psi1 = np.moveaxis(psi1, 0, target_qubit)
        psi1 = psi1.reshape(2**nqb)

        # watch out ordering. Expval unitary is cpx so psi1 on the right to match DM.
        assert np.allclose(np.dot(psi.conjugate(), psi1), dm.expectation_single(op, target_qubit))

    def test_tensor_fail(self) -> None:
        dm = DensityMatrix(nqubit=1)
        with pytest.raises(TypeError):
            dm.tensor("hello")
        with pytest.raises(TypeError):
            dm.tensor(1)

    @pytest.mark.parametrize("n", range(3))
    def test_tensor_without_data_success(self, n: int) -> None:
        dm_a = DensityMatrix(nqubit=n)
        dm_b = DensityMatrix(nqubit=n + 1)
        dm_a.tensor(dm_b)
        assert dm_a.Nqubit == 2 * n + 1
        assert dm_a.rho.shape == (2 ** (2 * n + 1), 2 ** (2 * n + 1))

    # TODO: Use pytest.mark.parametrize after refactoring randobj.rand_dm
    def test_tensor_with_data_success(self) -> None:
        for n in range(3):
            data_a = randobj.rand_dm(2**n, dm_dtype=False)
            dm_a = DensityMatrix(data=data_a)

            data_b = randobj.rand_dm(2 ** (n + 1), dm_dtype=False)
            dm_b = DensityMatrix(data=data_b)
            dm_a.tensor(dm_b)
            assert dm_a.Nqubit == 2 * n + 1
            assert dm_a.rho.shape == (2 ** (2 * n + 1), 2 ** (2 * n + 1))
            assert np.allclose(dm_a.rho, np.kron(data_a, data_b))

    def test_cnot_fail(self) -> None:
        dm = DensityMatrix(nqubit=2)
        with pytest.raises(ValueError):
            dm.cnot((1, 1))
        with pytest.raises(ValueError):
            dm.cnot((-1, 1))
        with pytest.raises(ValueError):
            dm.cnot((1, -1))
        with pytest.raises(ValueError):
            dm.cnot((1, 2))
        with pytest.raises(ValueError):
            dm.cnot((2, 1))

    def test_cnot_success(self, fx_rng: Generator) -> None:
        dm = DensityMatrix(nqubit=2)
        original_matrix = dm.rho.copy()
        dm.cnot((0, 1))
        expected_matrix = np.array([[1, 1, 1, 1] * 4]).reshape((4, 4)) / 4
        assert np.allclose(dm.rho, expected_matrix)
        dm.cnot((0, 1))
        assert np.allclose(dm.rho, original_matrix)

        # test on 2 qubits only
        psi = _randstate_raw(2, fx_rng)
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
        n = fx_rng.integers(2, 4)
        psi = _randstate_raw(n, fx_rng)
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
        assert np.allclose(rho, expected_matrix)

    def test_swap_fail(self) -> None:
        dm = DensityMatrix(nqubit=2)
        with pytest.raises(ValueError):
            dm.swap((1, 1))
        with pytest.raises(ValueError):
            dm.swap((-1, 1))
        with pytest.raises(ValueError):
            dm.swap((1, -1))
        with pytest.raises(ValueError):
            dm.swap((1, 2))
        with pytest.raises(ValueError):
            dm.swap((2, 1))

    def test_swap_success(self, fx_rng: Generator) -> None:
        dm = DensityMatrix(nqubit=2)
        original_matrix = dm.rho.copy()
        dm.swap((0, 1))
        expected_matrix = np.array([[1, 1, 1, 1] * 4]).reshape((4, 4)) / 4
        assert np.allclose(dm.rho, expected_matrix)
        dm.swap((0, 1))
        assert np.allclose(dm.rho, original_matrix)

        psi = _randstate_raw(2, fx_rng)
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

    def test_entangle_fail(self) -> None:
        dm = DensityMatrix(nqubit=3)
        with pytest.raises(ValueError):
            dm.entangle((1, 1))
        with pytest.raises(ValueError):
            dm.entangle((1, 3))
        with pytest.raises(ValueError):
            dm.entangle((0, 1, 2))

    def test_entangle_success(self, fx_rng: Generator) -> None:
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

        psi = _randstate_raw(2, fx_rng)
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2))
        dm = DensityMatrix(data=np.outer(psi, psi.conj()))
        edge = (0, 1)
        dm.entangle(edge)
        rho = dm.rho
        psi = psi.reshape((2, 2))
        psi = np.tensordot(CZ_TENSOR, psi, ((2, 3), edge))
        psi = np.moveaxis(psi, (0, 1), edge)
        expected_matrix = np.outer(psi, psi.conj())
        assert np.allclose(rho, expected_matrix)

    def test_evolve_success(self, fx_rng: Generator) -> None:
        # single-qubit gate
        # check against evolve_single

        nqubits = fx_rng.integers(2, 4)
        nqubits_op = 1

        # random statevector
        psi = _randstate_raw(nqubits, fx_rng)
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2))

        # density matrix calculation
        dm = DensityMatrix(data=np.outer(psi, psi.conj()))
        dm_single = deepcopy(dm)

        op = randobj.rand_unit(2**nqubits_op)
        i = fx_rng.integers(0, nqubits)

        # need a list format for a single target
        dm.evolve(op, [i])
        dm_single.evolve_single(op, i)

        assert np.allclose(dm.rho, dm_single.rho)

        # 2-qubit gate

        nqubits = fx_rng.integers(2, 4)
        nqubits_op = 2

        # random unitary
        op = randobj.rand_unit(2**nqubits_op)

        # random pair of indices
        edge = tuple(random.sample(range(nqubits), 2))

        # random statevector to compare to
        psi = _randstate_raw(nqubits, fx_rng)
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2))

        # density matrix calculation
        dm = DensityMatrix(data=np.outer(psi, psi.conj()))
        dm.evolve(op, edge)
        rho = dm.rho

        psi = psi.reshape((2,) * nqubits)
        psi = np.tensordot(op.reshape((2,) * 2 * nqubits_op), psi, ((2, 3), edge))
        psi = np.moveaxis(psi, (0, 1), edge)
        expected_matrix = np.outer(psi, psi.conj())
        assert np.allclose(rho, expected_matrix)

        # 3-qubit gate
        nqubits = fx_rng.integers(3, 5)
        nqubits_op = 3

        # random unitary
        op = randobj.rand_unit(2**nqubits_op)

        # 3 random indices
        targets = tuple(random.sample(range(nqubits), 3))

        # random statevector to compare to
        psi = _randstate_raw(nqubits, fx_rng)
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2))

        # density matrix calculation
        dm = DensityMatrix(data=np.outer(psi, psi.conj()))
        dm.evolve(op, targets)
        rho = dm.rho

        psi = psi.reshape((2,) * nqubits)
        psi = np.tensordot(op.reshape((2,) * 2 * nqubits_op), psi, ((3, 4, 5), targets))
        psi = np.moveaxis(psi, (0, 1, 2), targets)
        expected_matrix = np.outer(psi, psi.conj())
        assert np.allclose(rho, expected_matrix)

    def test_evolve_fail(self, fx_rng: Generator) -> None:
        # test on 3-qubit gate just in case.
        nqubits = fx_rng.integers(3, 5)
        nqubits_op = 3

        # random unitary
        op = randobj.rand_unit(2**nqubits_op)
        # 3 random indices
        dm = DensityMatrix(nqubit=nqubits)

        # dimension mismatch
        with pytest.raises(ValueError):
            dm.evolve(op, (1, 1))
        with pytest.raises(ValueError):
            dm.evolve(op, (0, 1, 2, 3))
        # incorrect range
        with pytest.raises(ValueError):
            dm.evolve(op, (-1, 0, 1))
        # repeated index
        with pytest.raises(ValueError):
            dm.evolve(op, (0, 1, 1))

        # check not square matrix
        with pytest.raises(ValueError):
            dm.evolve(fx_rng.uniform(size=(2, 3)), (0, 1))

        # check higher dimensional matrix
        with pytest.raises(ValueError):
            dm.evolve(fx_rng.uniform(size=(2, 2, 3)), (0, 1))

        # check square but with incorrect dimension (non-qubit type)
        with pytest.raises(ValueError):
            dm.evolve(fx_rng.uniform(size=(5, 5)), (0, 1))

    # TODO: the test for normalization is done at initialization with data.
    # Now check that all operations conserve the norm.
    def test_normalize(self, fx_rng: Generator) -> None:
        data = randobj.rand_dm(2 ** fx_rng.integers(2, 4), dm_dtype=False)

        dm = DensityMatrix(data / data.trace())
        dm.normalize()
        assert np.allclose(np.trace(dm.rho), 1)

    def test_ptrace_fail(self) -> None:
        dm = DensityMatrix(nqubit=0)
        with pytest.raises(AssertionError):
            dm.ptrace((0,))
        dm = DensityMatrix(nqubit=2)
        with pytest.raises(AssertionError):
            dm.ptrace((2,))

    def test_ptrace(self) -> None:
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
            [[1 / 3, 0, np.sqrt(2) / 3, 0], [0, 0, 0, 0], [np.sqrt(2) / 3, 0, 2 / 3, 0], [0, 0, 0, 0]],
        )
        assert np.allclose(dm.rho, expected_matrix)

    def test_apply_dephasing_channel(self, fx_rng: Generator) -> None:
        # check on single qubit first
        # # create random density matrix
        # data = randobj.rand_herm(2 ** fx_rng.integers(2, 4))
        dm = randobj.rand_dm(2)

        # copy of initial dm
        rho_test = deepcopy(dm.rho)

        # create dephasing channel
        prob = fx_rng.uniform()
        dephase_channel = dephasing_channel(prob)

        # useless since checked in apply_channel method.
        assert isinstance(dephase_channel, KrausChannel)
        # useless since checked in the constructor.
        assert dephase_channel.is_normalized()

        # apply channel. list with single element needed.
        # if Channel.nqubit == 1 use list with single element.
        dm.apply_channel(dephase_channel, [0])
        identity = np.array([[1.0, 0.0], [0.0, 1.0]])

        # compare
        expected_dm = (
            np.sqrt(1 - prob) ** 2 * identity @ rho_test @ identity.conj().T
            + np.sqrt(prob) ** 2 * Ops.z @ rho_test @ Ops.z.conj().T
        )

        assert np.allclose(expected_dm.trace(), 1.0)
        assert np.allclose(dm.rho, expected_dm)

        nqubits = fx_rng.integers(2, 5)

        i = fx_rng.integers(0, nqubits)

        # create random density matrix from statevector

        # random statevector to compare to
        psi = _randstate_raw(nqubits, fx_rng)
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2))

        # build DensityMatrix
        dm = DensityMatrix(data=np.outer(psi, psi.conj()))

        # create dephasing channel
        prob = fx_rng.uniform()
        dephase_channel = dephasing_channel(prob)

        # useless since checked in apply_channel method.
        assert isinstance(dephase_channel, KrausChannel)
        # useless since checked in the constructor.
        assert dephase_channel.is_normalized()

        # apply channel. list with single element needed.
        # if Channel.nqubit == 1 use list with single element.
        dm.apply_channel(dephase_channel, [i])

        # compute on the statevector
        # psi.reshape((2,) * nqubits)
        # tmp = np.zeros(psi.shape)

        identity = np.array([[1.0, 0.0], [0.0, 1.0]])

        # by hand: operator list and gate application
        psi_evolved = np.tensordot(identity, psi.reshape((2,) * nqubits), (1, i))
        psi_evolved = np.moveaxis(psi_evolved, 0, i)

        psi_evolvedb = np.tensordot(Ops.z, psi.reshape((2,) * nqubits), (1, i))
        psi_evolvedb = np.moveaxis(psi_evolvedb, 0, i)

        # compute final density matrix
        psi_evolved = np.reshape(psi_evolved, (2**nqubits))
        psi_evolvedb = np.reshape(psi_evolvedb, (2**nqubits))
        expected_dm = np.sqrt(1 - prob) ** 2 * np.outer(psi_evolved, psi_evolved.conj()) + np.sqrt(
            prob,
        ) ** 2 * np.outer(psi_evolvedb, psi_evolvedb.conj())

        # compare
        assert np.allclose(expected_dm.trace(), 1.0)
        assert np.allclose(dm.rho, expected_dm)

    def test_apply_depolarising_channel(self, fx_rng: Generator) -> None:
        # check on single qubit first
        # # create random density matrix
        # data = randobj.rand_herm(2 ** fx_rng.integers(2, 4))
        dm = randobj.rand_dm(2)

        # copy of initial dm
        rho_test = deepcopy(dm.rho)

        # create dephasing channel
        prob = fx_rng.uniform()
        depol_channel = depolarising_channel(prob)

        # useless since checked in apply_channel method.
        assert isinstance(depol_channel, KrausChannel)
        # useless since checked in the constructor.
        assert depol_channel.is_normalized()

        # apply channel. list with single element needed.
        # if Channel.nqubit == 1 use list with single element.
        dm.apply_channel(depol_channel, [0])
        identity = np.array([[1.0, 0.0], [0.0, 1.0]])

        # compare
        expected_dm = (
            np.sqrt(1 - prob) ** 2 * identity @ rho_test @ identity.conj().T
            + np.sqrt(prob / 3.0) ** 2 * Ops.x @ rho_test @ Ops.x.conj().T
            + np.sqrt(prob / 3.0) ** 2 * Ops.y @ rho_test @ Ops.y.conj().T
            + np.sqrt(prob / 3.0) ** 2 * Ops.z @ rho_test @ Ops.z.conj().T
        )

        assert np.allclose(expected_dm.trace(), 1.0)
        assert np.allclose(dm.rho, expected_dm)

        # chek against statevector backend by hand for now.
        # create random density matrix

        nqubits = fx_rng.integers(2, 5)

        # target qubit index
        i = fx_rng.integers(0, nqubits)

        # create random density matrix from statevector

        # random statevector to compare to
        psi = _randstate_raw(nqubits, fx_rng)
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2))

        # build DensityMatrix
        dm = DensityMatrix(data=np.outer(psi, psi.conj()))

        # create dephasing channel
        prob = fx_rng.uniform()
        depol_channel = depolarising_channel(prob)

        # useless since checked in apply_channel method.
        assert isinstance(depol_channel, KrausChannel)
        # useless since checked in the constructor.
        assert depol_channel.is_normalized()

        # apply channel. list with single element needed.
        # if Channel.nqubit == 1 use list with single element.
        dm.apply_channel(depol_channel, [i])

        # compute on the statevector
        # psi.reshape((2,) * nqubits)
        # tmp = np.zeros(psi.shape)

        identity = np.array([[1.0, 0.0], [0.0, 1.0]])

        # by hand: operator list and gate application
        psi_evolved = np.tensordot(identity, psi.reshape((2,) * nqubits), (1, i))
        psi_evolved = np.moveaxis(psi_evolved, 0, i)

        psi_evolvedb = np.tensordot(Ops.x, psi.reshape((2,) * nqubits), (1, i))
        psi_evolvedb = np.moveaxis(psi_evolvedb, 0, i)

        psi_evolvedc = np.tensordot(Ops.y, psi.reshape((2,) * nqubits), (1, i))
        psi_evolvedc = np.moveaxis(psi_evolvedc, 0, i)

        psi_evolvedd = np.tensordot(Ops.z, psi.reshape((2,) * nqubits), (1, i))
        psi_evolvedd = np.moveaxis(psi_evolvedd, 0, i)

        # compute final density matrix
        psi_evolved = np.reshape(psi_evolved, (2**nqubits))
        psi_evolvedb = np.reshape(psi_evolvedb, (2**nqubits))
        psi_evolvedc = np.reshape(psi_evolvedc, (2**nqubits))
        psi_evolvedd = np.reshape(psi_evolvedd, (2**nqubits))

        expected_dm = (
            np.sqrt(1 - prob) ** 2 * np.outer(psi_evolved, psi_evolved.conj())
            + np.sqrt(prob / 3.0) ** 2 * np.outer(psi_evolvedb, psi_evolvedb.conj())
            + np.sqrt(prob / 3.0) ** 2 * np.outer(psi_evolvedc, psi_evolvedc.conj())
            + np.sqrt(prob / 3.0) ** 2 * np.outer(psi_evolvedd, psi_evolvedd.conj())
        )

        # compare
        assert np.allclose(expected_dm.trace(), 1.0)
        assert np.allclose(dm.rho, expected_dm)

    def test_apply_random_channel_one_qubit(self, fx_rng: Generator) -> None:
        """
        test random 1-qubit channel.
        Especially checks for complex parameters.
        """

        # check against statevector backend by hand for now.
        # create random density matrix

        nqubits = fx_rng.integers(2, 5)
        # identity = np.array([[1.0, 0.0], [0.0, 1.0]])

        # target qubit index
        i = fx_rng.integers(0, nqubits)

        # create random density matrix from statevector

        # random statevector to compare to
        psi = _randstate_raw(nqubits, fx_rng)
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2))

        # build DensityMatrix
        dm = DensityMatrix(data=np.outer(psi, psi.conj()))

        # create random channel
        # random_channel utility already checked for type and such
        # here dim = 2 (single qubit) and rank is between 1 and 4
        nqb = 1
        dim = 2**nqb
        rk = int(fx_rng.integers(1, dim**2 + 1))
        channel = randobj.rand_channel_kraus(dim=dim, rank=rk)

        # apply channel. list with single element needed.
        # if Channel.nqubit == 1 use list with single element.
        dm.apply_channel(channel, [i])

        # compute on the statevector
        # psi.reshape((2,) * nqubits)
        # tmp = np.zeros(psi.shape)

        # initialize. NOT a DM object, just a matrix.
        expected_dm = np.zeros((2**nqubits, 2**nqubits), dtype=np.complex128)

        for elem in channel.kraus_ops:  # kraus_ops is a list of dicts
            psi_evolved = np.tensordot(elem["operator"], psi.reshape((2,) * nqubits), (1, i))
            psi_evolved = np.moveaxis(psi_evolved, 0, i)
            expected_dm += elem["coef"] * np.conj(elem["coef"]) * np.outer(psi_evolved, np.conj(psi_evolved))

        # compare
        assert np.allclose(expected_dm.trace(), 1.0)
        assert np.allclose(dm.rho, expected_dm)

    def test_apply_random_channel_two_qubits(self, fx_rng: Generator) -> None:
        """
        test random 2-qubit channel on a rank 1 dm (pure state). Generalizes by linearity.
        Especially checks for complex parameters.
        """

        nqubits = fx_rng.integers(2, 5)

        # target qubits indices
        qubits = tuple(random.sample(range(nqubits), 2))

        # create random density matrix from statevector
        # random statevector to compare to
        psi = _randstate_raw(nqubits, fx_rng)
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2))
        # build DensityMatrix
        dm = DensityMatrix(data=np.outer(psi, psi.conj()))

        # create random channel
        # for 2 qubits, rank between 1 and 16
        # number of qubits it acts on
        nqb = 2
        dim = 2**nqb
        rk = int(fx_rng.integers(1, dim**2 + 1))
        channel = randobj.rand_channel_kraus(dim=dim, rank=rk)

        dm.apply_channel(channel, qubits)

        # initialize. NOT a DM object, just a matrix.
        expected_dm = np.zeros((2**nqubits, 2**nqubits), dtype=np.complex128)
        # reshape statevec since not in tensor format
        for elem in channel.kraus_ops:  # kraus_ops is a list of dicts
            psi_evolved = np.tensordot(
                elem["operator"].reshape((2,) * 2 * nqb),
                psi.reshape((2,) * nqubits),
                ((2, 3), qubits),
            )
            psi_evolved = np.moveaxis(psi_evolved, (0, 1), qubits)
            expected_dm += elem["coef"] * np.conj(elem["coef"]) * np.outer(psi_evolved, np.conj(psi_evolved))

        assert np.allclose(expected_dm.trace(), 1.0)
        assert np.allclose(dm.rho, expected_dm)

    def test_apply_channel_fail(self, fx_rng: Generator) -> None:
        """
        test apply a channel that is not a Channel object
        """
        nqubits = fx_rng.integers(2, 5)
        i = fx_rng.integers(0, nqubits)

        psi = _randstate_raw(nqubits, fx_rng)
        psi /= np.sqrt(np.sum(np.abs(psi) ** 2))

        # build DensityMatrix
        dm = DensityMatrix(data=np.outer(psi, psi.conj()))

        with pytest.raises(TypeError):
            dm.apply_channel("a", [i])


class TestDensityMatrixBackend:
    """Test for DensityMatrixBackend class."""

    # test initialization only
    def test_init_success(self, fx_rng: Generator, hadamardpattern, randpattern, nqb) -> None:
        # plus state (default)
        backend = DensityMatrixBackend(hadamardpattern)
        dm = DensityMatrix(nqubit=1)
        assert np.allclose(dm.rho, backend.state.rho)
        # assert backend.state.Nqubit == 1
        assert backend.state.dims() == (2, 2)

        # minus state
        backend = DensityMatrixBackend(randpattern, input_state=graphix.states.BasicStates.MINUS)
        dm = DensityMatrix(nqubit=nqb, data=graphix.states.BasicStates.MINUS)
        assert np.allclose(dm.rho, backend.state.rho)
        # assert backend.state.Nqubit == 1
        assert backend.state.dims() == (2**nqb, 2**nqb)

        rand_angles = fx_rng.random(nqb) * 2 * np.pi
        rand_planes = fx_rng.choice(np.array([i for i in graphix.pauli.Plane]), nqb)
        states = [graphix.states.PlanarState(plane=i, angle=j) for i, j in zip(rand_planes, rand_angles)]

        expected_dm = DensityMatrix(data=states).rho

        backend = DensityMatrixBackend(randpattern, input_state=states)
        dm = backend.state

        assert dm.dims() == (2**nqb, 2**nqb)
        assert np.allclose(dm.rho, expected_dm)
        assert backend.Nqubit == nqb

    def test_init_fail(self, fx_rng: Generator, nqb, randpattern) -> None:
        rand_angles = fx_rng.random(nqb + 1) * 2 * np.pi
        rand_planes = fx_rng.choice(np.array([i for i in graphix.pauli.Plane]), nqb + 1)
        states = [graphix.states.PlanarState(plane=i, angle=j) for i, j in zip(rand_planes, rand_angles)]

        # test init from State Iterable with incorrect size
        with pytest.raises(ValueError):
            _backend = DensityMatrixBackend(randpattern, input_state=states)

        # don't provide required pattern argument
        with pytest.raises(TypeError):
            DensityMatrixBackend()

    def test_init_success_2(self) -> None:
        circ = Circuit(1)
        circ.rx(0, np.pi / 2)
        pattern = circ.transpile().pattern
        backend = DensityMatrixBackend(pattern)
        assert backend.pattern == pattern
        assert backend.results == pattern.results
        assert backend.node_index == [0]
        assert backend.Nqubit == 1
        assert backend.max_qubit_num == 12

    def test_add_nodes(self) -> None:
        circ = Circuit(1)
        pattern = circ.transpile().pattern
        backend = DensityMatrixBackend(pattern)
        backend.add_nodes([1])
        expected_matrix = np.array([0.25] * 16).reshape(4, 4)
        assert np.allclose(backend.state.rho, expected_matrix)

    def test_entangle_nodes(self) -> None:
        circ = Circuit(1)
        pattern = circ.transpile().pattern
        backend = DensityMatrixBackend(pattern)
        backend.add_nodes([1])
        backend.entangle_nodes((0, 1))
        expected_matrix = np.array([[1, 1, 1, -1], [1, 1, 1, -1], [1, 1, 1, -1], [-1, -1, -1, 1]]) / 4
        assert np.allclose(backend.state.rho, expected_matrix)

        backend.entangle_nodes((0, 1))
        assert np.allclose(backend.state.rho, np.array([0.25] * 16).reshape(4, 4))

    def test_measure(self) -> None:
        circ = Circuit(1)
        circ.rx(0, np.pi / 2)
        pattern = circ.transpile().pattern

        backend = DensityMatrixBackend(pattern)
        backend.add_nodes([1, 2])
        backend.entangle_nodes((0, 1))
        backend.entangle_nodes((1, 2))
        backend.measure(backend.pattern[-4])

        expected_matrix_1 = np.kron(np.array([[1, 0], [0, 0]]), np.ones((2, 2)) / 2)
        expected_matrix_2 = np.kron(np.array([[0, 0], [0, 1]]), np.array([[0.5, -0.5], [-0.5, 0.5]]))
        assert np.allclose(backend.state.rho, expected_matrix_1) or np.allclose(backend.state.rho, expected_matrix_2)

    def test_measure_pr_calc(self) -> None:
        # circuit there just to provide a measurement command to try out. Weird.
        circ = Circuit(1)
        circ.rx(0, np.pi / 2)
        pattern = circ.transpile().pattern

        backend = DensityMatrixBackend(pattern, pr_calc=True)
        backend.add_nodes([1, 2])
        backend.entangle_nodes((0, 1))
        backend.entangle_nodes((1, 2))
        backend.measure(backend.pattern[-4])

        # 3-qubit linear graph state: |+0+> + |-1->
        expected_matrix_1 = np.kron(np.array([[1, 0], [0, 0]]), np.ones((2, 2)) / 2)
        expected_matrix_2 = np.kron(np.array([[0, 0], [0, 1]]), np.array([[0.5, -0.5], [-0.5, 0.5]]))

        assert np.allclose(backend.state.rho, expected_matrix_1) or np.allclose(backend.state.rho, expected_matrix_2)

    def test_correct_byproduct(self) -> None:
        circ = Circuit(1)
        circ.rx(0, np.pi / 2)
        pattern = circ.transpile().pattern

        backend = DensityMatrixBackend(pattern)
        # node 0 initialized in Backend
        backend.add_nodes([1, 2])
        backend.entangle_nodes((0, 1))
        backend.entangle_nodes((1, 2))
        backend.measure(backend.pattern[-4])
        backend.measure(backend.pattern[-3])
        backend.correct_byproduct(backend.pattern[-2])
        backend.correct_byproduct(backend.pattern[-1])
        backend.finalize()
        rho = backend.state.rho

        backend = StatevectorBackend(pattern)
        # node 0 initialized in Backend
        backend.add_nodes([1, 2])
        backend.entangle_nodes((0, 1))
        backend.entangle_nodes((1, 2))
        backend.measure(backend.pattern[-4])
        backend.measure(backend.pattern[-3])
        backend.correct_byproduct(backend.pattern[-2])
        backend.correct_byproduct(backend.pattern[-1])
        backend.finalize()
        psi = backend.state.psi

        assert np.allclose(rho, np.outer(psi, psi.conj()))
