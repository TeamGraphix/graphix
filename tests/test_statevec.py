from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest
from numpy.random import Generator

from graphix.clifford import Clifford
from graphix.ops import Ops
from graphix.random_objects import rand_unit
from graphix.sim.statevec import Statevec
from graphix.states import BasicStates

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Literal

    from numpy.random import PCG64

    _ENCODING = Literal["LSB", "MSB"]

    from graphix.states import State


def generate_rnd_data(rng: Generator, nqubits: int) -> npt.NDArray[np.complex128]:
    length = 1 << nqubits
    data = rng.random(length) + 1j * rng.random(length)
    data /= np.sqrt(np.sum(np.abs(data) ** 2))
    return data


class TestInit:
    N_JUMPS = 3

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
        assert len(sv.psi) == 2**max_qubits

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


class TestSimulation:
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

    @pytest.mark.parametrize("jumps", range(1, 5))
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

    @pytest.mark.parametrize("jumps", range(1, 5))
    def test_expectation_value(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 4
        data = generate_rnd_data(rng, nqubits)
        sv = Statevec(data=data)
        op = rand_unit(4, rng)

        val_test = sv.expectation_value(op, (1, 2))
        val_ref = np.conjugate(data) @ functools.reduce(np.kron, (np.eye(2), op, np.eye(2))) @ data
        assert val_test == pytest.approx(val_ref)


# TODO: Refactor using parametrize
class TestFidelity:
    def test_fidelity_same_state(self) -> None:
        state = Statevec(data=BasicStates.PLUS)
        assert state.fidelity(state) == pytest.approx(1)

    def test_fidelity_orthogonal(self) -> None:
        zero = Statevec(data=BasicStates.ZERO)
        one = Statevec(data=BasicStates.ONE)
        assert zero.fidelity(one) == pytest.approx(0)

    def test_fidelity_known_value(self) -> None:
        # F(|0>, |+>) = 0.5
        zero = Statevec(data=BasicStates.ZERO)
        plus = Statevec(data=BasicStates.PLUS)
        assert zero.fidelity(plus) == pytest.approx(0.5)

    def test_fidelity_global_phase(self) -> None:
        plus = Statevec(data=BasicStates.PLUS)
        plus_rotated = Statevec(data=np.array([1, 1]) / np.sqrt(2) * 1j)
        assert plus.fidelity(plus_rotated) == pytest.approx(1)

    def test_fidelity_symmetry(self, fx_rng: Generator) -> None:
        length = 4
        vec_a = fx_rng.random(length) + 1j * fx_rng.random(length)
        vec_a /= np.sqrt(np.sum(np.abs(vec_a) ** 2))
        vec_b = fx_rng.random(length) + 1j * fx_rng.random(length)
        vec_b /= np.sqrt(np.sum(np.abs(vec_b) ** 2))
        a = Statevec(data=vec_a)
        b = Statevec(data=vec_b)
        assert a.fidelity(b) == pytest.approx(b.fidelity(a))

    def test_isclose_same_state(self) -> None:
        state = Statevec(data=BasicStates.PLUS)
        assert state.isclose(state)

    def test_isclose_orthogonal(self) -> None:
        zero = Statevec(data=BasicStates.ZERO)
        one = Statevec(data=BasicStates.ONE)
        assert not zero.isclose(one)

    def test_isclose_global_phase(self) -> None:
        plus = Statevec(data=BasicStates.PLUS)
        rotated = Statevec(data=np.array([1, 1]) / np.sqrt(2) * np.exp(1j * 0.7))
        assert plus.isclose(rotated)

    def test_isclose_tolerance(self) -> None:
        zero = Statevec(data=BasicStates.ZERO)
        almost = Statevec(data=np.array([np.sqrt(1 - 1e-8), np.sqrt(1e-8)]))
        assert not zero.isclose(almost)
        assert zero.isclose(almost, atol=1e-6)


@pytest.mark.parametrize(
    ("encoding", "dict_ref"),
    [
        ("LSB", {"000": 0.5, "010": 0.5, "100": -0.5, "110": -0.5}),
        ("MSB", {"000": 0.5, "010": 0.5, "001": -0.5, "011": -0.5}),
    ],
)
def test_to_dict(encoding: _ENCODING, dict_ref: Mapping[str, float]) -> None:
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
def test_to_prob_dict(encoding: _ENCODING, dict_ref: Mapping[str, float]) -> None:
    sv = Statevec(data=[BasicStates.ONE, BasicStates.PLUS, BasicStates.MINUS])
    for ket, amp2 in sv.to_prob_dict(encoding=encoding).items():
        assert np.isclose(dict_ref[ket], amp2.real)
        assert np.isclose(0, amp2.imag)


# class TestStatevecLegacy:
#     """Tests in this class compare the result against the existing statevector simulator in Graphix. They are not self-contained."""

#     N_JUMPS = 3

#     @pytest.mark.parametrize("jumps", range(1, N_JUMPS))
#     def test_entangle(self, fx_bg: PCG64, jumps: int) -> None:
#         rng = Generator(fx_bg.jumped(jumps))
#         nqubits = 5
#         sv_test = Statevec(generate_rnd_data(rng, nqubits))
#         sv_ref = SVLegacy(data=sv_test.flatten())
#         edge: tuple[int, int] = tuple(rng.choice(range(nqubits), size=2, replace=False))
#         for sv in [sv_test, sv_ref]:
#             sv.entangle(edge)

#         assert sv_ref.isclose(SVLegacy(data=sv_test.flatten()))

#     @pytest.mark.parametrize("jumps", range(1, N_JUMPS))
#     def test_swap(self, fx_bg: PCG64, jumps: int) -> None:
#         rng = Generator(fx_bg.jumped(jumps))
#         nqubits = 5
#         sv_test = Statevec(generate_rnd_data(rng, nqubits))
#         sv_ref = SVLegacy(data=sv_test.flatten())
#         edge: tuple[int, int] = tuple(rng.choice(range(nqubits), size=2, replace=False))
#         for sv in [sv_test, sv_ref]:
#             sv.swap(edge)

#         assert sv_ref.isclose(SVLegacy(data=sv_test.flatten()))

#     def test_evolve_single(self, fx_rng: Generator) -> None:
#         nqubits = 5
#         for clifford in Clifford:
#             sv_test = Statevec(generate_rnd_data(fx_rng, nqubits))
#             sv_ref = SVLegacy(data=sv_test.flatten())
#             qubit = int(fx_rng.integers(0, nqubits))
#             for sv in [sv_test, sv_ref]:
#                 sv.evolve_single(clifford.matrix, qubit)
#             assert sv_ref.isclose(SVLegacy(data=sv_test.flatten()))

#     def test_expectation_single(self, fx_rng: Generator) -> None:
#         nqubits = 5
#         for clifford in Clifford:
#             sv_test = Statevec(generate_rnd_data(fx_rng, nqubits))
#             sv_ref = SVLegacy(data=sv_test.flatten())
#             qubit = int(fx_rng.integers(0, nqubits))

#             val_test = sv_test.expectation_single(clifford.matrix, qubit)
#             val_ref = sv_ref.expectation_single(clifford.matrix, qubit)

#             assert math.isclose(val_test.real, val_ref.real, abs_tol=1e-12)
#             assert math.isclose(val_test.imag, val_ref.imag, abs_tol=1e-12)

#     def test_add_nodes(self, fx_rng: Generator) -> None:

#         max_qubits = 5
#         sv_test = Statevec(nqubit=0, max_qubits=max_qubits)
#         sv_ref = SVLegacy(nqubit=0)

#         for _ in range(max_qubits):  # Add a node at each iteration
#             data = generate_rnd_data(fx_rng, nqubits=1)
#             sv_test.add_nodes(1, data)
#             sv_ref.add_nodes(1, data)

#             assert sv_ref.isclose(SVLegacy(data=sv_test.flatten()))

#     @pytest.mark.parametrize(
#         "projector", [np.array([[1, 0], [0, 0]], dtype=np.complex128), np.array([[0, 0], [0, 1]], dtype=np.complex128)]
#     )
#     def test_remove_nodes(self, fx_rng: Generator, projector: npt.NDArray[np.complex128]) -> None:

#         nqubits = 5
#         sv_test = Statevec(generate_rnd_data(fx_rng, nqubits))
#         sv_ref = SVLegacy(data=sv_test.flatten())
#         q = 0
#         for _ in range(nqubits - 1):  # Remove a node at each iteration
#             sv_test.evolve_single(projector, q)
#             sv_test.remove_qubit(q)
#             sv_ref.evolve_single(projector, q)
#             sv_ref.remove_qubit(q)

#             assert sv_ref.isclose(SVLegacy(data=sv_test.flatten()))


# @pytest.mark.parametrize("jumps", range(1, 6))
# def test_pattern_simulator(fx_bg: PCG64, jumps: int) -> None:
#     rng = Generator(fx_bg.jumped(jumps))

#     nqubits = 5

#     pattern = rand_circuit(nqubits, depth=5, rng=rng).transpile().pattern
#     pattern.remove_pauli_measurements()

#     backend = StatevectorBackend.with_capacity(pattern.max_space())
#     sv_test = pattern.simulate_pattern(backend=backend, rng=rng)
#     sv_ref = pattern.simulate_pattern(backend=SBLegacy(), rng=rng)

#     assert sv_ref.isclose(SVLegacy(data=sv_test.flatten()))


# from __future__ import annotations

# import functools
# from typing import TYPE_CHECKING

# import numpy as np
# import pytest

# from graphix.fundamentals import ANGLE_PI, Plane
# from graphix.pattern import Pattern
# from graphix.sim.statevec import Statevec, _norm_numeric
# from graphix.states import BasicStates, PlanarState

# if TYPE_CHECKING:
#     from collections.abc import Mapping
#     from typing import Literal

#     from numpy.random import Generator

#     _ENCODING = Literal["LSB", "MSB"]


# class TestStatevec:
#     """Test for Statevec class. Particularly new constructor."""

#     # test initializing one qubit in plus state
#     def test_default_success(self) -> None:
#         vec = Statevec(nqubit=1)
#         assert np.allclose(vec.psi, np.array([1, 1] / np.sqrt(2)))
#         assert len(vec.dims()) == 1

#     def test_basicstates_success(self) -> None:
#         # minus
#         vec = Statevec(nqubit=1, data=BasicStates.MINUS)
#         assert np.allclose(vec.psi, np.array([1, -1] / np.sqrt(2)))
#         assert len(vec.dims()) == 1

#         # zero
#         vec = Statevec(nqubit=1, data=BasicStates.ZERO)
#         assert np.allclose(vec.psi, np.array([1, 0]), rtol=0, atol=1e-15)
#         assert len(vec.dims()) == 1

#         # one
#         vec = Statevec(nqubit=1, data=BasicStates.ONE)
#         assert np.allclose(vec.psi, np.array([0, 1]), rtol=0, atol=1e-15)
#         assert len(vec.dims()) == 1

#         # plus_i
#         vec = Statevec(nqubit=1, data=BasicStates.PLUS_I)
#         assert np.allclose(vec.psi, np.array([1, 1j] / np.sqrt(2)))
#         assert len(vec.dims()) == 1

#         # minus_i
#         vec = Statevec(nqubit=1, data=BasicStates.MINUS_I)
#         assert np.allclose(vec.psi, np.array([1, -1j] / np.sqrt(2)))
#         assert len(vec.dims()) == 1

#     # even more tests?
#     def test_default_tensor_success(self, fx_rng: Generator) -> None:
#         nqb = int(fx_rng.integers(2, 5))
#         print(f"nqb is {nqb}")
#         vec = Statevec(nqubit=nqb)
#         assert np.allclose(vec.psi, np.ones((2,) * nqb) / (np.sqrt(2)) ** nqb)
#         assert len(vec.dims()) == nqb

#         vec = Statevec(nqubit=nqb, data=BasicStates.MINUS_I)
#         sv_list = [BasicStates.MINUS_I.to_statevector() for _ in range(nqb)]
#         sv = functools.reduce(lambda a, b: np.kron(a, b).astype(np.complex128, copy=False), sv_list)
#         assert np.allclose(vec.psi, sv.reshape((2,) * nqb))
#         assert len(vec.dims()) == nqb

#         # tensor of same state
#         rand_angle = fx_rng.random() * 2 * ANGLE_PI
#         rand_plane = fx_rng.choice(np.array(Plane))
#         state = PlanarState(rand_plane, rand_angle)
#         vec = Statevec(nqubit=nqb, data=state)
#         sv_list = [state.to_statevector() for _ in range(nqb)]
#         sv = functools.reduce(lambda a, b: np.kron(a, b).astype(np.complex128, copy=False), sv_list)
#         assert np.allclose(vec.psi, sv.reshape((2,) * nqb))
#         assert len(vec.dims()) == nqb

#         # tensor of different states
#         rand_angles = fx_rng.random(nqb) * 2 * ANGLE_PI
#         rand_planes = fx_rng.choice(np.array(Plane), nqb)
#         states = [PlanarState(plane=i, angle=j) for i, j in zip(rand_planes, rand_angles, strict=True)]
#         vec = Statevec(nqubit=nqb, data=states)
#         sv_list = [state.to_statevector() for state in states]
#         sv = functools.reduce(lambda a, b: np.kron(a, b).astype(np.complex128, copy=False), sv_list)
#         assert np.allclose(vec.psi, sv.reshape((2,) * nqb))
#         assert len(vec.dims()) == nqb

#     def test_data_success(self, fx_rng: Generator) -> None:
#         nqb = fx_rng.integers(2, 5)
#         length = 2**nqb
#         rand_vec = fx_rng.random(length) + 1j * fx_rng.random(length)
#         rand_vec /= np.sqrt(np.sum(np.abs(rand_vec) ** 2))
#         vec = Statevec(data=rand_vec)
#         assert np.allclose(vec.psi, rand_vec.reshape((2,) * nqb))
#         assert len(vec.dims()) == nqb

#     # fail: incorrect len
#     def test_data_dim_fail(self, fx_rng: Generator) -> None:
#         length = 5
#         rand_vec = fx_rng.random(length) + 1j * fx_rng.random(length)
#         rand_vec /= np.sqrt(np.sum(np.abs(rand_vec) ** 2))
#         with pytest.raises(ValueError):
#             _vec = Statevec(data=rand_vec)

#     # fail: with less qubit than number of qubits inferred from a correct state vect
#     def test_data_dim_fail_mismatch(self, fx_rng: Generator) -> None:
#         nqb = 3
#         rand_vec = fx_rng.random(2**nqb) + 1j * fx_rng.random(2**nqb)
#         rand_vec /= np.sqrt(np.sum(np.abs(rand_vec) ** 2))
#         with pytest.raises(ValueError):
#             _vec = Statevec(nqubit=2, data=rand_vec)

#     # fail: not normalized
#     def test_data_norm_fail(self, fx_rng: Generator) -> None:
#         nqb = fx_rng.integers(2, 5)
#         length = 2**nqb
#         rand_vec = fx_rng.random(length) + 1j * fx_rng.random(length)
#         with pytest.raises(ValueError):
#             _vec = Statevec(data=rand_vec)

#     def test_defaults_to_one(self) -> None:
#         vec = Statevec()
#         assert len(vec.dims()) == 1

#     # try copying Statevec input
#     def test_copy_success(self, fx_rng: Generator) -> None:
#         nqb = fx_rng.integers(2, 5)
#         length = 2**nqb
#         rand_vec = fx_rng.random(length) + 1j * fx_rng.random(length)
#         rand_vec /= np.sqrt(np.sum(np.abs(rand_vec) ** 2))
#         test_vec = Statevec(data=rand_vec)
#         # try to copy it
#         vec = Statevec(data=test_vec)

#         assert np.allclose(vec.psi, test_vec.psi)
#         assert len(vec.dims()) == len(test_vec.dims())

#     # try calling with incorrect number of qubits compared to inferred one
#     def test_copy_fail(self, fx_rng: Generator) -> None:
#         nqb = int(fx_rng.integers(2, 5))
#         length = 1 << nqb
#         rand_vec = fx_rng.random(length) + 1j * fx_rng.random(length)
#         rand_vec /= np.sqrt(np.sum(np.abs(rand_vec) ** 2))
#         test_vec = Statevec(data=rand_vec)

#         with pytest.raises(ValueError):
#             _vec = Statevec(nqubit=length - 1, data=test_vec)

#     def test_nqubits(self) -> None:
#         for i in [1, 2, 5]:
#             sv = Statevec(nqubit=i)
#             assert sv.nqubit == i

#     def test_nqubits_pattern(self) -> None:
#         p = Pattern(input_nodes=[0, 1, 2])
#         sv = p.simulate_pattern(backend="statevector")
#         assert sv.nqubit == 3


# def test_normalize() -> None:
#     statevec = Statevec(nqubit=1, data=BasicStates.PLUS)
#     statevec.remove_qubit(0)
#     assert _norm_numeric(statevec.psi.astype(np.complex128, copy=False)) == 1
