from __future__ import annotations

import copy
import itertools
import sys
import typing
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.random import PCG64, Generator

from graphix import clifford
from graphix.command import C, E, M, N, X, Z
from graphix.pattern import CommandNode, Pattern, shift_outcomes
from graphix.pauli import Plane
from graphix.random_objects import rand_circuit, rand_gate
from graphix.sim.density_matrix import DensityMatrix
from graphix.simulator import PatternSimulator
from graphix.states import PlanarState
from graphix.transpiler import Circuit

if TYPE_CHECKING:
    import collections.abc
    from collections.abc import Sequence

    from graphix.sim.base_backend import Backend
    from graphix.sim.statevec import Statevec


def compare_backend_result_with_statevec(backend: str, backend_state, statevec: Statevec) -> float:
    if backend == "statevector":
        return np.abs(np.dot(backend_state.flatten().conjugate(), statevec.flatten()))
    elif backend == "densitymatrix":
        return np.abs(np.dot(backend_state.rho.flatten().conjugate(), DensityMatrix(statevec).rho.flatten()))
    else:
        raise NotImplementedError(backend)


Outcome = typing.Literal[0, 1]


class IterGenerator:
    def __init__(self, it: collections.abc.Iterable[Outcome]) -> None:
        self.__it = iter(it)

    def choice(self, _outcomes: list[Outcome]) -> Outcome:
        return next(self.__it)


class TestPattern:
    # this fails without behaviour modification
    def test_manual_generation(self) -> None:
        pattern = Pattern()
        pattern.add(N(node=0))
        pattern.add(N(node=1))
        pattern.add(M(node=0))

    def test_standardize(self, fx_rng: Generator) -> None:
        nqubits = 2
        depth = 1
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern
        pattern.standardize(method="global")
        assert pattern.is_standard()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_minimize_space(self, fx_rng: Generator) -> None:
        nqubits = 5
        depth = 5
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern
        pattern.standardize(method="global")
        pattern.minimize_space()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_minimize_space_with_gflow(self, fx_bg: PCG64, jumps: int, use_rustworkx: bool = True) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 3
        depth = 3
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rand_gate(nqubits, depth, pairs, rng)
        pattern = circuit.transpile().pattern
        pattern.standardize(method="global")
        pattern.shift_signals(method="global")
        pattern.perform_pauli_measurements(use_rustworkx=use_rustworkx)
        pattern.minimize_space()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    @pytest.mark.filterwarnings("ignore:Simulating using densitymatrix backend with no noise.")
    @pytest.mark.parametrize("backend_type", ["statevector", "densitymatrix", "tensornetwork"])
    def test_empty_output_nodes(
        self, backend_type: typing.Literal["statevector", "densitymatrix", "tensornetwork"]
    ) -> None:
        pattern = Pattern(input_nodes=[0])
        pattern.add(M(node=0, angle=0.5))

        def simulate_and_measure() -> int:
            sim = PatternSimulator(pattern, backend_type)
            sim.run()
            if backend_type == "statevector":
                assert sim.backend.state.dims() == ()
            elif backend_type == "densitymatrix":
                assert sim.backend.state.dims() == (1, 1)
            elif backend_type == "tensornetwork":
                assert sim.backend.state.to_statevector().shape == (1,)
            return sim.measure_method.results[0]

        nb_shots = 1000
        nb_ones = sum(1 for _ in range(nb_shots) if simulate_and_measure())
        assert abs(nb_ones - nb_shots / 2) < nb_shots / 10

    def test_minimize_space_graph_maxspace_with_flow(self, fx_rng: Generator) -> None:
        max_qubits = 20
        for nqubits in range(2, max_qubits):
            depth = 5
            pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
            circuit = rand_gate(nqubits, depth, pairs, fx_rng)
            pattern = circuit.transpile().pattern
            pattern.standardize(method="global")
            pattern.minimize_space()
            assert pattern.max_space() == nqubits + 1

    def test_parallelize_pattern(self, fx_rng: Generator) -> None:
        nqubits = 2
        depth = 1
        circuit = rand_circuit(nqubits, depth, fx_rng)
        pattern = circuit.transpile().pattern
        pattern.standardize(method="global")
        pattern.parallelize_pattern()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_shift_signals(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 2
        depth = 1
        circuit = rand_circuit(nqubits, depth, rng)
        pattern = circuit.transpile().pattern
        pattern.standardize(method="global")
        pattern.shift_signals(method="global")
        assert pattern.is_standard()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    @pytest.mark.parametrize("backend", ["statevector", "densitymatrix"])
    # TODO: tensor network backend is excluded because "parallel preparation strategy does not support not-standardized pattern".
    def test_pauli_measurement_random_circuit(
        self, fx_bg: PCG64, jumps: int, backend: Backend, use_rustworkx: bool = True
    ) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 3
        depth = 3
        circuit = rand_circuit(nqubits, depth, rng)
        pattern = circuit.transpile().pattern
        pattern.standardize(method="global")
        pattern.shift_signals(method="global")
        pattern.perform_pauli_measurements(use_rustworkx=use_rustworkx)
        pattern.minimize_space()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(backend, rng=rng)
        assert compare_backend_result_with_statevec(backend, state_mbqc, state) == pytest.approx(1)

    @pytest.mark.parametrize("plane", Plane)
    @pytest.mark.parametrize("angle", [0.0, 0.5, 1.0, 1.5])
    def test_pauli_measurement_single(self, plane: Plane, angle: float, use_rustworkx: bool = True) -> None:
        pattern = Pattern(input_nodes=[0, 1])
        pattern.add(E(nodes=[0, 1]))
        pattern.add(M(node=0, plane=plane, angle=angle))
        pattern_ref = pattern.copy()
        pattern.perform_pauli_measurements(use_rustworkx=use_rustworkx)
        state = pattern.simulate_pattern()
        state_ref = pattern_ref.simulate_pattern(pr_calc=False, rng=IterGenerator([0]))
        assert np.abs(np.dot(state.flatten().conjugate(), state_ref.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_pauli_measurement_leave_input_random_circuit(
        self, fx_bg: PCG64, jumps: int, use_rustworkx: bool = True
    ) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 3
        depth = 3
        circuit = rand_circuit(nqubits, depth, rng)
        pattern = circuit.transpile().pattern
        pattern.standardize(method="global")
        pattern.shift_signals(method="global")
        pattern.perform_pauli_measurements(use_rustworkx=use_rustworkx, leave_input=True)
        pattern.minimize_space()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_pauli_measurement_opt_gate(self, fx_bg: PCG64, jumps: int, use_rustworkx: bool = True) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 3
        depth = 3
        circuit = rand_circuit(nqubits, depth, rng, use_rzz=True)
        pattern = circuit.transpile(opt=True).pattern
        pattern.standardize(method="global")
        pattern.shift_signals(method="global")
        pattern.perform_pauli_measurements(use_rustworkx=use_rustworkx)
        pattern.minimize_space()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_pauli_measurement_opt_gate_transpiler(self, fx_bg: PCG64, jumps: int, use_rustworkx: bool = True) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 3
        depth = 3
        circuit = rand_circuit(nqubits, depth, rng, use_rzz=True)
        pattern = circuit.standardize_and_transpile(opt=True).pattern
        pattern.standardize(method="global")
        pattern.shift_signals(method="global")
        pattern.perform_pauli_measurements(use_rustworkx=use_rustworkx)
        pattern.minimize_space()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_pauli_measurement_opt_gate_transpiler_without_signalshift(
        self,
        fx_bg: PCG64,
        jumps: int,
        use_rustworkx: bool = True,
    ) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 3
        depth = 3
        circuit = rand_circuit(nqubits, depth, rng, use_rzz=True)
        pattern = circuit.standardize_and_transpile(opt=True).pattern
        pattern.perform_pauli_measurements(use_rustworkx=use_rustworkx)
        pattern.minimize_space()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern(rng=rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize(
        "use_rustworkx",
        [
            False,
            pytest.param(
                True,
                marks=pytest.mark.skipif(sys.modules.get("rustworkx") is None, reason="rustworkx not installed"),
            ),
        ],
    )
    def test_pauli_measurement(self, use_rustworkx: bool) -> None:
        # test pattern is obtained from 3-qubit QFT with pauli measurement
        circuit = Circuit(3)
        for i in range(3):
            circuit.h(i)
        circuit.x(1)
        circuit.x(2)

        # QFT
        circuit.h(2)
        cp(circuit, np.pi / 4, 0, 2)
        cp(circuit, np.pi / 2, 1, 2)
        circuit.h(1)
        cp(circuit, np.pi / 2, 0, 1)
        circuit.h(0)
        swap(circuit, 0, 2)

        pattern = circuit.transpile().pattern
        pattern.standardize(method="global")
        pattern.shift_signals(method="global")
        pattern.perform_pauli_measurements(use_rustworkx=use_rustworkx)

        isolated_nodes = pattern.get_isolated_nodes()
        # 48-node is the isolated and output node.
        isolated_nodes_ref = {48}

        assert isolated_nodes == isolated_nodes_ref

    @pytest.mark.parametrize(
        "use_rustworkx",
        [
            False,
            pytest.param(
                True,
                marks=pytest.mark.skipif(sys.modules.get("rustworkx") is None, reason="rustworkx not installed"),
            ),
        ],
    )
    def test_pauli_measurement_leave_input(self, use_rustworkx: bool) -> None:
        # test pattern is obtained from 3-qubit QFT with pauli measurement
        circuit = Circuit(3)
        for i in range(3):
            circuit.h(i)
        circuit.x(1)
        circuit.x(2)

        # QFT
        circuit.h(2)
        cp(circuit, np.pi / 4, 0, 2)
        cp(circuit, np.pi / 2, 1, 2)
        circuit.h(1)
        cp(circuit, np.pi / 2, 0, 1)
        circuit.h(0)
        swap(circuit, 0, 2)

        pattern = circuit.transpile().pattern
        pattern.standardize(method="global")
        pattern.shift_signals(method="global")
        pattern.perform_pauli_measurements(use_rustworkx=use_rustworkx, leave_input=True)

        isolated_nodes = pattern.get_isolated_nodes()
        # There is no isolated node.
        isolated_nodes_ref = set()

        assert isolated_nodes == isolated_nodes_ref

    def test_get_meas_plane(self) -> None:
        preset_meas_plane = [
            Plane.XY,
            Plane.XY,
            Plane.XY,
            Plane.YZ,
            Plane.YZ,
            Plane.YZ,
            Plane.XZ,
            Plane.XZ,
            Plane.XZ,
        ]
        vop_list = [0, 5, 6]  # [identity, S gate, H gate]
        pattern = Pattern(input_nodes=list(range(len(preset_meas_plane))))
        for i in range(len(preset_meas_plane)):
            pattern.add(M(node=i, plane=preset_meas_plane[i]).clifford(clifford.get(vop_list[i % 3])))
        ref_meas_plane = {
            0: Plane.XY,
            1: Plane.XY,
            2: Plane.YZ,
            3: Plane.YZ,
            4: Plane.XZ,
            5: Plane.XY,
            6: Plane.XZ,
            7: Plane.YZ,
            8: Plane.XZ,
        }
        meas_plane = pattern.get_meas_plane()
        assert meas_plane == ref_meas_plane

    @pytest.mark.parametrize("plane", Plane)
    @pytest.mark.parametrize("method", ["local", "global", "direct"])
    def test_shift_signals_plane(self, plane: Plane, method: str) -> None:
        pattern = Pattern(input_nodes=[0])
        for i in (1, 2, 3):
            pattern.add(N(node=i))
            pattern.add(E(nodes=[0, i]))
        pattern.add(M(node=0, angle=0.5))
        pattern.add(M(node=1, angle=0.5))
        pattern.add(M(node=2, angle=0.5, plane=plane, s_domain=[0], t_domain=[1]))
        pattern.add(Z(node=3, domain=[2]))
        pattern_ref = copy.deepcopy(pattern)
        pattern.standardize(method="global")
        signal_dict = pattern.shift_signals(method=method)
        # Test for every possible outcome of each measure
        for outcomes_ref in itertools.product(*([[0, 1]] * 3)):
            state_ref = pattern_ref.simulate_pattern(pr_calc=False, rng=IterGenerator(iter(outcomes_ref)))
            outcomes_p = shift_outcomes(dict(enumerate(outcomes_ref)), signal_dict)
            state_p = pattern.simulate_pattern(
                pr_calc=False, rng=IterGenerator(outcomes_p[i] for i in range(len(outcomes_p)))
            )
            assert np.abs(np.dot(state_p.flatten().conjugate(), state_ref.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_standardize_direct(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 5
        depth = 4
        circuit = rand_circuit(nqubits, depth, rng)
        pattern = circuit.transpile().pattern
        pattern.standardize(method="direct")
        assert pattern.is_standard()
        pattern.minimize_space()
        state_p = pattern.simulate_pattern()
        state_ref = circuit.simulate_statevector().statevec
        assert np.abs(np.dot(state_p.flatten().conjugate(), state_ref.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_shift_signals_direct(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 5
        depth = 4
        circuit = rand_circuit(nqubits, depth, rng)
        pattern = circuit.transpile().pattern
        pattern.standardize()
        pattern.shift_signals(method="direct")
        pattern.minimize_space()
        state_p = pattern.simulate_pattern()
        state_ref = circuit.simulate_statevector().statevec
        assert np.abs(np.dot(state_p.flatten().conjugate(), state_ref.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    @pytest.mark.parametrize("method", ["global", "direct"])
    def test_pauli_measurement_then_standardize(
        self, fx_bg: PCG64, jumps: int, method: str, use_rustworkx: bool = True
    ) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 3
        depth = 3
        circuit = rand_circuit(nqubits, depth, rng)
        pattern = circuit.transpile().pattern
        pattern.perform_pauli_measurements(use_rustworkx=use_rustworkx)
        pattern.standardize(method=method)
        pattern.minimize_space()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern.simulate_pattern()
        assert compare_backend_result_with_statevec("statevector", state_mbqc, state) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_standardize_two_cliffords(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        c0, c1 = rng.integers(len(clifford.TABLE), size=2)
        pattern = Pattern(input_nodes=[0])
        pattern.add(C(node=0, clifford=clifford.TABLE[c0]))
        pattern.add(C(node=0, clifford=clifford.TABLE[c1]))
        pattern_ref = pattern.copy()
        pattern.standardize(method="direct")
        state_ref = pattern_ref.simulate_pattern()
        state_p = pattern.simulate_pattern()
        assert np.abs(np.dot(state_p.flatten().conjugate(), state_ref.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 48))
    def test_standardize_domains_and_clifford(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        x, z = rng.integers(2, size=2)
        c = rng.integers(len(clifford.TABLE))
        pattern = Pattern(input_nodes=[0])
        pattern.results[1] = x
        pattern.add(X(node=0, domain={1}))
        pattern.results[2] = z
        pattern.add(Z(node=0, domain={2}))
        pattern.add(C(node=0, clifford=clifford.TABLE[c]))
        pattern_ref = pattern.copy()
        pattern.standardize(method="direct")
        state_ref = pattern_ref.simulate_pattern()
        state_p = pattern.simulate_pattern()
        assert np.abs(np.dot(state_p.flatten().conjugate(), state_ref.flatten())) == pytest.approx(1)


def cp(circuit: Circuit, theta: float, control: int, target: int) -> None:
    """Controlled rotation gate, decomposed."""  # noqa: D401
    circuit.rz(control, theta / 2)
    circuit.rz(target, theta / 2)
    circuit.cnot(control, target)
    circuit.rz(target, -1 * theta / 2)
    circuit.cnot(control, target)


def swap(circuit: Circuit, a: int, b: int) -> None:
    """Swap gate, decomposed."""
    circuit.cnot(a, b)
    circuit.cnot(b, a)
    circuit.cnot(a, b)


class TestLocalPattern:
    @pytest.mark.parametrize(
        "test",
        [
            ((0, 1), (0, 1), True),
            ((1, 0), (0, 1), True),
            ((0, 1), (2, 0), False),
            ((0, 1), (2, 3), False),
            ((1, 3), (4, 1), False),
        ],
    )
    def test_assert_equal_edge(self, test: tuple[tuple[int, int], tuple[int, int], bool]) -> None:
        assert assert_equal_edge(test[0], test[1]) == test[2]

    def test_no_gate(self) -> None:
        n = 3
        circuit = Circuit(n)
        pattern = circuit.transpile().pattern
        localpattern = pattern.get_local_pattern()
        for node in localpattern.nodes.values():
            assert node.seq == []

    def test_get_graph(self, fx_rng: Generator) -> None:
        nqubits = 5
        depth = 4
        pairs = [(i, np.mod(i + 1, nqubits)) for i in range(nqubits)]
        circuit = rand_gate(nqubits, depth, pairs, fx_rng)
        pattern = circuit.transpile().pattern
        nodes_ref, edges_ref = pattern.get_graph()

        localpattern = pattern.get_local_pattern()
        nodes, edges = localpattern.get_graph()

        # nodes check
        nodes_check1 = True
        nodes_check2 = True
        for node in nodes:
            if node not in nodes_ref:
                nodes_check1 = False
        for node in nodes_ref:
            if node not in nodes:
                nodes_check2 = False
        assert nodes_check1
        assert nodes_check2

        # edges check
        edges_check1 = True
        edges_check2 = True
        for edge in edges:
            edge_match = False
            for edge_ref in edges_ref:
                edge_match |= assert_equal_edge(edge, edge_ref)
            if not edge_match:
                edges_check1 = False
        for edge in edges_ref:
            edge_match = False
            for edge_ref in edges:
                edge_match |= assert_equal_edge(edge, edge_ref)
            if not edge_match:
                edges_check2 = False
        assert edges_check1
        assert edges_check2

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_standardize(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 5
        depth = 4
        circuit = rand_circuit(nqubits, depth, rng)
        pattern = circuit.transpile().pattern
        localpattern = pattern.get_local_pattern()
        localpattern.standardize()
        pattern = localpattern.get_pattern()
        assert pattern.is_standard()
        pattern.minimize_space()
        state_p = pattern.simulate_pattern(rng=rng)
        state_ref = circuit.simulate_statevector().statevec
        assert np.abs(np.dot(state_p.flatten().conjugate(), state_ref.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_shift_signals(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 5
        depth = 4
        circuit = rand_circuit(nqubits, depth, rng)
        pattern = circuit.transpile().pattern
        localpattern = pattern.get_local_pattern()
        localpattern.standardize()
        localpattern.shift_signals()
        pattern = localpattern.get_pattern()
        assert pattern.is_standard()
        pattern.minimize_space()
        state_p = pattern.simulate_pattern(rng=rng)
        state_ref = circuit.simulate_statevector().statevec
        assert np.abs(np.dot(state_p.flatten().conjugate(), state_ref.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_standardize_and_shift_signals(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 5
        depth = 4
        circuit = rand_circuit(nqubits, depth, rng)
        pattern = circuit.transpile().pattern
        pattern.standardize_and_shift_signals()
        assert pattern.is_standard()
        pattern.minimize_space()
        state_p = pattern.simulate_pattern(rng=rng)
        state_ref = circuit.simulate_statevector().statevec
        assert np.abs(np.dot(state_p.flatten().conjugate(), state_ref.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 4))
    def test_mixed_pattern_operations(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        processes = [
            [["standardize", "global"], ["standardize", "local"]],
            [["standardize", "local"], ["signal", "global"], ["signal", "local"]],
            [
                ["standardize", "local"],
                ["signal", "global"],
                ["standardize", "global"],
                ["signal", "local"],
            ],
        ]
        nqubits = 3
        depth = 2
        circuit = rand_circuit(nqubits, depth, rng)
        state_ref = circuit.simulate_statevector().statevec
        for process in processes:
            pattern = circuit.transpile().pattern
            for operation in process:
                if operation[0] == "standardize":
                    pattern.standardize(method=operation[1])
                elif operation[0] == "signal":
                    pattern.shift_signals(method=operation[1])
            assert pattern.is_standard()
            pattern.minimize_space()
            state_p = pattern.simulate_pattern(rng=rng)
            assert np.abs(np.dot(state_p.flatten().conjugate(), state_ref.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_opt_transpile_standardize(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 5
        depth = 4
        circuit = rand_circuit(nqubits, depth, rng)
        pattern = circuit.transpile(opt=True).pattern
        pattern.standardize(method="local")
        assert pattern.is_standard()
        pattern.minimize_space()
        state_p = pattern.simulate_pattern(rng=rng)
        state_ref = circuit.simulate_statevector().statevec
        assert np.abs(np.dot(state_p.flatten().conjugate(), state_ref.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_opt_transpile_shift_signals(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 5
        depth = 4
        circuit = rand_circuit(nqubits, depth, rng)
        pattern = circuit.transpile(opt=True).pattern
        pattern.standardize(method="local")
        pattern.shift_signals(method="local")
        assert pattern.is_standard()
        pattern.minimize_space()
        state_p = pattern.simulate_pattern(rng=rng)
        state_ref = circuit.simulate_statevector().statevec
        assert np.abs(np.dot(state_p.flatten().conjugate(), state_ref.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize(
        "test",
        [
            ([1, 2, 3, -1], True),
            ([1, 2, 3, -2, -3, -2, -4], True),
            ([1, -4, 2, -3, -1, 3], False),
            ([1, 2, 3, -1, -4, 2], False),
        ],
    )
    def test_node_is_standardized(self, test: tuple[list[int], bool]) -> None:
        seq, ref = test
        node = CommandNode(0, seq, [], [], False, [], [])
        result = node.is_standard()
        assert result == ref

    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_localpattern_is_standard(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 5
        depth = 4
        circuit = rand_circuit(nqubits, depth, rng)
        localpattern = circuit.transpile().pattern.get_local_pattern()
        result1 = localpattern.is_standard()
        localpattern.standardize()
        result2 = localpattern.is_standard()
        assert not result1
        assert result2

    def test_pauli_measurement_end_with_measure(self) -> None:
        # https://github.com/TeamGraphix/graphix/issues/153
        p = Pattern(input_nodes=[0])
        p.add(N(node=1))
        p.add(M(node=1, plane=Plane.XY))
        p.perform_pauli_measurements()

    @pytest.mark.parametrize("backend", ["statevector", "densitymatrix"])
    def test_arbitrary_inputs(self, fx_rng: Generator, nqb: int, rand_circ: Circuit, backend: str) -> None:
        rand_angles = fx_rng.random(nqb) * 2 * np.pi
        rand_planes = fx_rng.choice(np.array([i for i in Plane]), nqb)
        states = [PlanarState(plane=i, angle=j) for i, j in zip(rand_planes, rand_angles)]
        randpattern = rand_circ.transpile().pattern
        out = randpattern.simulate_pattern(backend=backend, input_state=states, rng=fx_rng)
        out_circ = rand_circ.simulate_statevector(input_state=states).statevec
        assert compare_backend_result_with_statevec(backend, out, out_circ) == pytest.approx(1)

    def test_arbitrary_inputs_tn(self, fx_rng: Generator, nqb: int, rand_circ: Circuit) -> None:
        rand_angles = fx_rng.random(nqb) * 2 * np.pi
        rand_planes = fx_rng.choice(np.array([i for i in Plane]), nqb)
        states = [PlanarState(plane=i, angle=j) for i, j in zip(rand_planes, rand_angles)]
        randpattern = rand_circ.transpile().pattern
        with pytest.raises(NotImplementedError):
            randpattern.simulate_pattern(
                backend="tensornetwork", graph_prep="sequential", input_state=states, rng=fx_rng
            )


def assert_equal_edge(edge: Sequence[int], ref: Sequence[int]) -> bool:
    return any(all(ei == ri for ei, ri in zip(edge, other)) for other in (ref, reversed(ref)))
