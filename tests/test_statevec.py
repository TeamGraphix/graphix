from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest
from numpy.random import Generator

from graphix.branch_selector import ConstBranchSelector
from graphix.clifford import Clifford
from graphix.instruction import Instruction
from graphix.measurements import Measurement
from graphix.ops import Ops
from graphix.pauli import Pauli
from graphix.random_objects import rand_unit
from graphix.sim.statevec import Statevec, StatevectorBackend
from graphix.states import BasicStates
from graphix.transpiler import Circuit

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Literal

    from numpy.random import PCG64

    _ENCODING = Literal["LSB", "MSB"]

    from graphix.states import PlanarState, State

N_JUMPS = 3


def generate_rnd_data(rng: Generator, nqubits: int) -> npt.NDArray[np.complex128]:
    length = 1 << nqubits
    data = rng.random(length) + 1j * rng.random(length)
    data /= np.sqrt(np.sum(np.abs(data) ** 2))
    return data


class TestStatevec:
    @pytest.mark.parametrize(
        ("state", "data_ref"),
        [
            (BasicStates.PLUS, np.array([1, 1] / np.sqrt(2))),
            (BasicStates.MINUS, np.array([1, -1] / np.sqrt(2))),
            (BasicStates.ZERO, np.array([1, 0])),
            (BasicStates.ONE, np.array([0, 1])),
            (BasicStates.PLUS_I, np.array([1, 1j] / np.sqrt(2))),
            (BasicStates.MINUS_I, np.array([1, -1j] / np.sqrt(2))),
        ],
    )
    def test_init_basic_states(self, state: State, data_ref: npt.NDArray[np.complex128]) -> None:
        sv = Statevec(data=state)
        assert np.allclose(sv.flatten(), data_ref)

    @pytest.mark.parametrize("nqubit", range(5))
    def test_init_random_state(self, fx_rng: Generator, nqubit: int) -> None:
        data = generate_rnd_data(fx_rng, nqubit)
        sv = Statevec(data)
        assert np.allclose(sv.flatten(), data)

    def test_init_preallocation(self) -> None:
        nqubit = 2
        max_qubits = 5
        sv = Statevec(data=[BasicStates.PLUS, BasicStates.ZERO], nqubit=nqubit, max_qubits=max_qubits)

        assert np.allclose(sv.flatten(), np.array([1, 0, 1, 0]) / np.sqrt(2))
        assert sv.nqubit == nqubit
        assert sv.max_qubits == max_qubits
        assert len(sv._psi) == 2**max_qubits

    @pytest.mark.parametrize("nqubit", range(5))
    def test_init_statevec(self, fx_rng: Generator, nqubit: int) -> None:
        data = generate_rnd_data(fx_rng, nqubit)
        sv_1 = Statevec(data)
        sv_2 = Statevec(sv_1)
        assert np.allclose(sv_1.psi, sv_2.psi)
        assert sv_1.nqubit == sv_2.nqubit
        assert sv_1.max_qubits == sv_2.max_qubits

    @pytest.mark.parametrize("nqubit", range(5))
    def test_init_statevec_max_qubits(self, fx_rng: Generator, nqubit: int) -> None:
        data = generate_rnd_data(fx_rng, nqubit)
        sv_1 = Statevec(data)
        sv_2 = Statevec(data=sv_1, max_qubits=nqubit + 1)
        assert np.allclose(sv_1.flatten(), sv_2.flatten())
        assert sv_1.nqubit == sv_2.nqubit
        assert sv_1.max_qubits + 1 == sv_2.max_qubits

    # fail: length data is not a power of 2
    @pytest.mark.parametrize("length", [3, 5, 6, 7])
    def test_init_dim_fail(self, fx_rng: Generator, length: int) -> None:
        rand_vec = fx_rng.random(length) + 1j * fx_rng.random(length)
        rand_vec /= np.sqrt(np.sum(np.abs(rand_vec) ** 2))
        with pytest.raises(ValueError):
            Statevec(data=rand_vec)

    # fail: different qubit than number of qubits inferred from data
    @pytest.mark.parametrize("nqubit", [2, 4])
    def test_init_dim_mismatch_fail(self, fx_rng: Generator, nqubit: int) -> None:
        data = generate_rnd_data(fx_rng, nqubit)
        with pytest.raises(ValueError):
            Statevec(nqubit=3, data=data)

    # fail: not normalized
    def test_init_norm_fail(self, fx_rng: Generator) -> None:
        data = 5 * generate_rnd_data(fx_rng, 3)
        with pytest.raises(ValueError):
            Statevec(data=data)

    # fail: max qubits smaller than number of qubits
    def test_init_max_qubits_fail(self) -> None:
        nqubit = 4
        max_qubits = 3
        with pytest.raises(ValueError):
            Statevec(nqubit=nqubit, max_qubits=max_qubits)

    # fail: incorrect number of qubits
    @pytest.mark.parametrize("nqubit", range(5))
    def test_init_statevec_fail(self, fx_rng: Generator, nqubit: int) -> None:
        data = generate_rnd_data(fx_rng, nqubit)
        sv_1 = Statevec(data)
        with pytest.raises(ValueError):
            Statevec(data=sv_1, nqubit=nqubit + 1)

    # fail: incorrect max_qubits qubits
    @pytest.mark.parametrize("nqubit", range(5))
    def test_init_statevec_max_qubits_fail(self, fx_rng: Generator, nqubit: int) -> None:
        data = generate_rnd_data(fx_rng, nqubit)
        sv_1 = Statevec(data=data, max_qubits=6)
        with pytest.raises(ValueError):
            Statevec(data=sv_1, max_qubits=nqubit + 1)

    @pytest.mark.parametrize(
        ("sv", "edge", "data_ref"),
        [
            (Statevec(data=BasicStates.ZERO, nqubit=2), (0, 1), np.array([1, 0, 0, 0])),
            (Statevec(data=[BasicStates.PLUS, BasicStates.PLUS]), (0, 1), np.array([1, 1, 1, -1]) / 2),
            (Statevec(data=[BasicStates.ONE, BasicStates.MINUS]), (0, 1), np.array([0, 0, 1, 1]) / np.sqrt(2)),
            (
                Statevec(data=np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2)),
                (0, 2),
                np.array([1, 0, 0, 0, 0, 0, 0, -1]) / np.sqrt(2),
            ),
        ],
    )
    def test_entangle(self, sv: Statevec, edge: tuple[int, int], data_ref: npt.NDArray[np.complex128]) -> None:
        sv.entangle(edge)
        assert np.allclose(sv.flatten(), data_ref)

    @pytest.mark.parametrize(
        ("sv", "q", "op", "data_ref"),
        [
            (Statevec(data=BasicStates.ZERO, nqubit=2), 0, Clifford.X.matrix, np.array([0, 0, 1, 0])),
            (
                Statevec(data=[BasicStates.PLUS, BasicStates.PLUS]),
                1,
                Clifford.H.matrix,
                np.array([1, 0, 1, 0]) / np.sqrt(2),
            ),
            (
                Statevec(data=[BasicStates.PLUS, BasicStates.MINUS]),
                0,
                np.array([[1, 0], [0, np.exp(0.25j * np.pi)]]),
                np.array([1, -1, np.exp(0.25j * np.pi), -np.exp(0.25j * np.pi)]) / 2,
            ),
            (
                Statevec(data=np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2)),
                1,
                Clifford.Z.matrix,
                np.array([1, 0, 0, 0, 0, 0, 0, -1]) / np.sqrt(2),
            ),
        ],
    )
    def test_evolve_single(
        self, sv: Statevec, q: int, op: npt.NDArray[np.complex128], data_ref: npt.NDArray[np.complex128]
    ) -> None:
        sv.evolve_single(op, q)
        assert np.allclose(sv.flatten(), data_ref)

    @pytest.mark.parametrize(
        ("sv", "q", "op", "exp_ref"),
        [
            (Statevec(data=BasicStates.ZERO, nqubit=2), 0, Clifford.X.matrix, 0),
            (Statevec(data=[BasicStates.PLUS, BasicStates.PLUS]), 1, Clifford.H.matrix, 1 / np.sqrt(2)),
            (
                Statevec(data=[BasicStates.PLUS, BasicStates.MINUS]),
                0,
                np.array([[1, 0], [0, np.exp(0.25j * np.pi)]]),
                (1 + np.exp(0.25j * np.pi)) / 2,
            ),
            (
                Statevec(data=np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2)),
                1,
                Clifford.Z.matrix,
                0,
            ),
        ],
    )
    def test_expectation_single(
        self, sv: Statevec, q: int, op: npt.NDArray[np.complex128], exp_ref: np.complex128
    ) -> None:
        assert np.isclose(sv.expectation_single(op, q), exp_ref)

    def test_add_nodes(self, fx_rng: Generator) -> None:
        max_qubits = 5
        sv_test = Statevec(nqubit=0, max_qubits=max_qubits)
        psi_ref = np.array([1.0 + 0.0j])

        for _ in range(max_qubits):  # Add a node at each iteration
            data = generate_rnd_data(fx_rng, nqubits=1)
            psi_ref = np.kron(psi_ref, data)
            sv_test.add_nodes(1, data)
            assert np.allclose(sv_test.flatten(), psi_ref)

    def test_add_nodes_beyond_max_qubits(self, fx_rng: Generator) -> None:
        max_qubits = 3
        sv_test = Statevec(nqubit=max_qubits, max_qubits=max_qubits)

        # We create a reference to `psi` to ensure that array is not extended with np.ndarray.resize
        psi_ref = sv_test.psi  # noqa: F841

        nqubits_new = 2
        data = generate_rnd_data(fx_rng, nqubits=nqubits_new)
        psi = np.kron(sv_test.psi, data)

        sv_test.add_nodes(nqubits_new, data)
        assert np.allclose(sv_test.flatten(), psi)

    @pytest.mark.parametrize(
        ("sv", "q", "sv_ref"),
        [
            (Statevec(data=BasicStates.ZERO, nqubit=2), 0, Statevec(data=BasicStates.ZERO, nqubit=1)),
            (Statevec(data=[BasicStates.PLUS, BasicStates.PLUS]), 1, Statevec(data=BasicStates.PLUS, nqubit=1)),
            (Statevec(data=[BasicStates.PLUS, BasicStates.MINUS]), 0, Statevec(data=BasicStates.MINUS, nqubit=1)),
            (Statevec(data=[BasicStates.ZERO, BasicStates.ONE]), 0, Statevec(data=BasicStates.ONE, nqubit=1)),
            # In previous testcase, branch 1 is 0 (psi_10 == psi_11 == 0), and first element of branch 0 is 0 too (psi_00 == 0)!
            (
                Statevec(data=[BasicStates.PLUS_I, BasicStates.ONE, BasicStates.PLUS]),
                1,
                Statevec(data=[BasicStates.PLUS_I, BasicStates.PLUS], nqubit=2),
            ),
        ],
    )
    def test_remove_qubit(self, sv: Statevec, q: int, sv_ref: Statevec) -> None:
        sv.remove_qubit(q)
        assert np.allclose(sv.flatten(), sv_ref.flatten())

    @pytest.mark.parametrize(
        "state",
        [
            BasicStates.PLUS,
            BasicStates.MINUS,
            BasicStates.ZERO,
            BasicStates.ONE,
            BasicStates.PLUS_I,
            BasicStates.MINUS_I,
        ],
    )
    def test_measurement_into_each_xyz_basis(self, state: PlanarState) -> None:
        n = 3
        k = 0
        statevector = state.to_statevector()
        m_op = np.outer(statevector, statevector.T.conjugate())
        sv = Statevec(nqubit=n)
        sv.evolve_single(m_op, k)

        if state is BasicStates.MINUS:
            # Measurement into |-> results in a 0-norm vector
            with pytest.raises(RuntimeError):
                sv.remove_qubit(k)
        else:
            sv.remove_qubit(k)
            sv2 = Statevec(nqubit=n - 1)
            assert sv.isclose(sv2)

    @pytest.mark.parametrize(
        ("sv", "qargs", "op", "data_ref"),
        [
            (Statevec(data=BasicStates.ZERO, nqubit=2), (0,), Clifford.X.matrix, np.array([0, 0, 1, 0])),
            (
                Statevec(data=[BasicStates.PLUS, BasicStates.PLUS]),
                (1,),
                Clifford.H.matrix,
                np.array([1, 0, 1, 0]) / np.sqrt(2),
            ),
            (
                Statevec(data=[BasicStates.PLUS, BasicStates.MINUS]),
                (0,),
                np.array([[1, 0], [0, np.exp(0.25j * np.pi)]]),
                np.array([1, -1, np.exp(0.25j * np.pi), -np.exp(0.25j * np.pi)]) / 2,
            ),
            (
                Statevec(data=np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2)),
                (1,),
                Clifford.Z.matrix,
                np.array([1, 0, 0, 0, 0, 0, 0, -1]) / np.sqrt(2),
            ),
            (
                Statevec(data=np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2)),
                (0, 1),
                Ops.CNOT,
                np.array([1, 0, 0, 0, 0, 1, 0, 0]) / np.sqrt(2),
            ),
            (
                Statevec(data=np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2)),
                (0, 2),
                Ops.CNOT,
                np.array([1, 0, 0, 0, 0, 0, 1, 0]) / np.sqrt(2),
            ),
            (
                Statevec(data=np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2), max_qubits=5),
                (1, 2),
                Ops.CNOT,
                np.array([1, 0, 0, 0, 0, 0, 1, 0]) / np.sqrt(2),
            ),
            (
                Statevec(data=np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2), max_qubits=5),
                (0, 1, 2),
                Ops.CCX,
                np.array([1, 0, 0, 0, 0, 0, 1, 0]) / np.sqrt(2),
            ),
        ],
    )
    def test_evolve(
        self, sv: Statevec, qargs: tuple[int, ...], op: npt.NDArray[np.complex128], data_ref: npt.NDArray[np.complex128]
    ) -> None:
        sv.evolve(op, qargs)
        assert np.allclose(sv.flatten(), data_ref)

    @pytest.mark.parametrize("jumps", range(1, N_JUMPS))
    def test_evolve_rnd(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 4
        data = generate_rnd_data(rng, nqubits)
        sv = Statevec(data=data)
        op = rand_unit(8, rng)

        sv.evolve(op, (1, 2, 3))
        data_ref = np.kron(np.eye(2), op) @ data
        sv_ref = Statevec(data_ref)

        assert sv.isclose(sv_ref)

    @pytest.mark.parametrize("jumps", range(1, N_JUMPS))
    def test_expectation_value(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 4
        data = generate_rnd_data(rng, nqubits)
        sv = Statevec(data=data)
        op = rand_unit(4, rng)

        val_test = sv.expectation_value(op, (1, 2))
        val_ref = np.conjugate(data) @ functools.reduce(np.kron, (np.eye(2), op, np.eye(2))) @ data
        assert val_test == pytest.approx(val_ref)

    @pytest.mark.parametrize(
        ("sv1", "sv2", "fidelity"),
        [
            (Statevec(data=BasicStates.PLUS), Statevec(data=BasicStates.PLUS), 1),
            (Statevec(data=BasicStates.ZERO), Statevec(data=BasicStates.ONE), 0),
            (Statevec(data=BasicStates.ZERO), Statevec(data=BasicStates.PLUS), 0.5),
            (Statevec(data=BasicStates.PLUS), Statevec(data=np.array([1, 1]) / np.sqrt(2) * 1j), 1),
        ],
    )
    def test_fidelity(self, sv1: Statevec, sv2: Statevec, fidelity: float) -> None:
        assert sv1.fidelity(sv2) == pytest.approx(fidelity)

    @pytest.mark.parametrize(
        ("sv1", "sv2", "isclose", "atol"),
        [
            (Statevec(data=BasicStates.PLUS), Statevec(data=BasicStates.PLUS), True, 0.0),
            (Statevec(data=BasicStates.ZERO), Statevec(data=BasicStates.ONE), False, 0.0),
            (
                Statevec(data=BasicStates.PLUS),
                Statevec(data=np.array([1, 1]) / np.sqrt(2) * np.exp(1j * 0.7)),
                True,
                0.0,
            ),
            (Statevec(data=BasicStates.ZERO), Statevec(data=np.array([np.sqrt(1 - 1e-8), np.sqrt(1e-8)])), False, 0.0),
            (
                Statevec(data=BasicStates.ZERO),
                Statevec(data=np.array([np.sqrt(1 - 1e-8), np.sqrt(1e-8)])),
                True,
                1.0e-6,
            ),
        ],
    )
    def test_isclose(self, sv1: Statevec, sv2: Statevec, isclose: bool, atol: float) -> None:
        if isclose:
            assert sv1.isclose(sv2, atol=atol)
        else:
            assert not sv1.isclose(sv2, atol=atol)

    @pytest.mark.parametrize(
        ("encoding", "dict_ref"),
        [
            ("LSB", {"000": 0.5, "010": 0.5, "100": -0.5, "110": -0.5}),
            ("MSB", {"000": 0.5, "010": 0.5, "001": -0.5, "011": -0.5}),
        ],
    )
    def test_to_dict(self, encoding: _ENCODING, dict_ref: Mapping[str, float]) -> None:
        sv = Statevec(data=[BasicStates.ZERO, BasicStates.PLUS, BasicStates.MINUS])
        for ket, amp in sv.to_dict(encoding=encoding).items():
            assert np.isclose(dict_ref[ket], amp.real)
            assert np.isclose(0, amp.imag)

    @pytest.mark.parametrize(
        ("encoding", "dict_ref"),
        [
            ("LSB", {"001": 0.25, "011": 0.25, "101": 0.25, "111": 0.25}),
            ("MSB", {"100": 0.25, "110": 0.25, "101": 0.25, "111": 0.25}),
        ],
    )
    def test_to_prob_dict(self, encoding: _ENCODING, dict_ref: Mapping[str, float]) -> None:
        sv = Statevec(data=[BasicStates.ONE, BasicStates.PLUS, BasicStates.MINUS])
        for ket, amp2 in sv.to_prob_dict(encoding=encoding).items():
            assert np.isclose(dict_ref[ket], amp2.real)
            assert np.isclose(0, amp2.imag)

    def test_simulation_identity(self, fx_rng: Generator) -> None:
        nqubits = 3
        qc = Circuit(nqubits)
        angle_1 = 2 * fx_rng.random()
        angle_2 = 2 * fx_rng.random()
        qc.extend(
            [
                Instruction.H(0),
                Instruction.CNOT(0, 1),
                Instruction.CNOT(1, 2),
                Instruction.RZZ(0, 1, angle_1),
                Instruction.RY(2, angle_2),
            ]
        )
        qc.extend(
            [
                Instruction.RZZ(0, 1, -angle_1),
                Instruction.RY(2, -angle_2),
                Instruction.CNOT(1, 2),
                Instruction.CNOT(0, 1),
                Instruction.H(0),
            ]
        )
        pattern = qc.transpile().pattern
        data = generate_rnd_data(fx_rng, nqubits)
        sv = Statevec(data=data)
        sv_test = pattern.simulate_pattern(backend="statevector", input_state=sv, rng=fx_rng)

        assert sv_test.isclose(sv)


class TestStatevectorBackend:
    @pytest.mark.parametrize(
        ("state", "data_ref"),
        [
            (BasicStates.PLUS, np.array([1, 1] / np.sqrt(2))),
            (BasicStates.MINUS, np.array([1, -1] / np.sqrt(2))),
            (BasicStates.ZERO, np.array([1, 0])),
            (BasicStates.ONE, np.array([0, 1])),
            (BasicStates.PLUS_I, np.array([1, 1j] / np.sqrt(2))),
            (BasicStates.MINUS_I, np.array([1, -1j] / np.sqrt(2))),
        ],
    )
    def test_init_basic_states(self, state: State, data_ref: npt.NDArray[np.complex128]) -> None:
        backend = StatevectorBackend()
        backend.add_nodes([0], data=state)
        sv = Statevec(data=data_ref)
        assert backend.state.isclose(sv)

    @pytest.mark.parametrize(
        ("n_nodes"),
        range(5),
    )
    def test_init_capacity(self, n_nodes: int, fx_rng: Generator) -> None:
        capacity = 3
        data = generate_rnd_data(fx_rng, n_nodes)
        backend = StatevectorBackend.with_capacity(capacity)
        backend.add_nodes(range(n_nodes), data=data)
        sv = Statevec(data=data)
        assert backend.state.isclose(sv)
        assert backend.state.max_qubits == max(n_nodes, capacity)

    def test_init_branch_selector(self) -> None:
        max_qubits = 5
        sv = Statevec(nqubit=3)
        bs = ConstBranchSelector(result=0)

        backend = StatevectorBackend.with_capacity(max_qubits, sv, branch_selector=bs)

        assert backend.branch_selector is bs
        assert backend.state.isclose(sv)
        assert backend.state.max_qubits == 5

    def test_init_branch_selector_kw(self) -> None:
        sv = Statevec(nqubit=3)
        bs = ConstBranchSelector(result=0)

        with pytest.raises(TypeError):
            # Branch selector is keyword only (enforced by dataclass sentinel)
            StatevectorBackend(sv, bs)  # type: ignore[misc, arg-type]

        backend = StatevectorBackend(sv, branch_selector=bs)

        assert backend.branch_selector is bs
        assert backend.state.isclose(sv)
        assert backend.state.max_qubits == 3

    @pytest.mark.parametrize(
        ("clifford"),
        Clifford,
    )
    def test_clifford(self, clifford: Clifford, fx_rng: Generator) -> None:
        nqubits = 4
        data = generate_rnd_data(fx_rng, nqubits=nqubits)

        backend = StatevectorBackend()
        backend.add_nodes(nodes=range(nqubits), data=data)
        backend.apply_clifford(node=0, clifford=clifford)

        vec = Statevec(nqubit=nqubits, data=data)
        vec.evolve_single(clifford.matrix, 0)

        assert backend.state.isclose(vec)

    @pytest.mark.parametrize("jumps", range(1, N_JUMPS))
    def test_deterministic_measure_0(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        # plus state & zero state (default), but with tossed coins

        backend = StatevectorBackend()
        coins = [rng.choice([0, 1]), rng.choice([0, 1])]
        expected_result = sum(coins) % 2
        states = [
            Pauli.X.eigenstate(coins[0]),
            Pauli.Z.eigenstate(coins[1]),
        ]
        nodes = range(len(states))
        backend.add_nodes(nodes=nodes, data=states)
        backend.entangle_nodes(edge=(nodes[0], nodes[1]))
        measurement = Measurement.X
        node_to_measure = backend.node_index[0]
        result = backend.measure(node=node_to_measure, measurement=measurement, rng=rng)
        assert result == expected_result

    @pytest.mark.parametrize("jumps", range(1, N_JUMPS))
    def test_deterministic_measure_1(self, fx_bg: PCG64, jumps: int) -> None:
        """Entangle |+> state with N |0> states, the (XY,0) measurement yields the outcome 0 with probability 1."""
        rng = Generator(fx_bg.jumped(jumps))
        n_nodes = 11
        backend = StatevectorBackend.with_capacity(n_nodes)
        states = [BasicStates.PLUS, *(BasicStates.ZERO for _ in range(n_nodes - 1))]
        backend.add_nodes(nodes=range(n_nodes), data=states)
        for i in range(1, n_nodes):
            backend.entangle_nodes(edge=(0, i))
        measurement = Measurement.X
        node_to_measure = backend.node_index[0]
        result = backend.measure(node=node_to_measure, measurement=measurement, rng=rng)
        assert result == 0
        assert list(backend.node_index) == list(range(1, n_nodes))

    @pytest.mark.parametrize("jumps", range(1, N_JUMPS))
    def test_deterministic_measure_many(self, fx_bg: PCG64, jumps: int) -> None:
        """Entangle |+> state with N |0> states, the (XY,0) measurement yields the outcome 0 with probability 1."""
        rng = Generator(fx_bg.jumped(jumps))
        # plus state (default)
        backend = StatevectorBackend()
        n_traps = 5
        n_neighbors = 5
        n_others = 5
        traps = [Pauli.X.eigenstate() for _ in range(n_traps)]
        dummies = [Pauli.Z.eigenstate() for _ in range(n_neighbors)]
        others = [Pauli.I.eigenstate() for _ in range(n_others)]
        states = traps + dummies + others
        nodes = range(len(states))
        backend.add_nodes(nodes=nodes, data=states)

        for dummy in nodes[n_traps : n_traps + n_neighbors]:
            for trap in nodes[:n_traps]:
                backend.entangle_nodes(edge=(trap, dummy))
            for other in nodes[n_traps + n_neighbors :]:
                backend.entangle_nodes(edge=(other, dummy))

        # Same measurement for all traps
        measurement = Measurement.X

        for trap in nodes[:n_traps]:
            node_to_measure = trap
            result = backend.measure(node=node_to_measure, measurement=measurement, rng=rng)
            assert result == 0

        assert list(backend.node_index) == list(range(n_traps, n_neighbors + n_traps + n_others))

    @pytest.mark.parametrize("jumps", range(1, N_JUMPS))
    def test_deterministic_measure_with_coin(self, fx_bg: PCG64, jumps: int) -> None:
        """Entangle |+> state with N |0> states, the (XY,0) measurement yields the outcome 0 with probability 1.

        We add coin toss to that.
        """
        rng = Generator(fx_bg.jumped(jumps))
        # plus state (default)
        backend = StatevectorBackend()
        n_neighbors = 10
        coins = [rng.choice([0, 1])] + [rng.choice([0, 1]) for _ in range(n_neighbors)]
        expected_result = sum(coins) % 2
        states = [Pauli.X.eigenstate(coins[0])] + [Pauli.Z.eigenstate(coins[i + 1]) for i in range(n_neighbors)]
        nodes = range(len(states))
        backend.add_nodes(nodes=nodes, data=states)

        for i in range(1, n_neighbors + 1):
            backend.entangle_nodes(edge=(nodes[0], i))
        measurement = Measurement.X
        node_to_measure = backend.node_index[0]
        result = backend.measure(node=node_to_measure, measurement=measurement, rng=rng)
        assert result == expected_result
        assert list(backend.node_index) == list(range(1, n_neighbors + 1))
