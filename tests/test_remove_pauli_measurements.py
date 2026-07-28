from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import pytest
from numpy.random import Generator

from graphix import (
    ANGLE_PI,
    Axis,
    BlochMeasurement,
    Circuit,
    Clifford,
    Command,
    Measurement,
    OpenGraph,
    Pattern,
    PauliMeasurement,
    Sign,
    StandardizedPattern,
)
from graphix.random_objects import rand_circuit, rand_state_vector
from graphix.remove_pauli_measurements import PauliPushingCut, _RemovePauliMeasurements, remove_pauli_measurements

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from collections.abc import Set as AbstractSet

    from numpy.random import PCG64

    from graphix.command import Node
    from graphix.remove_pauli_measurements import Graph


def opengraph_lemma_2_31(measurements: Mapping[Node, Measurement]) -> OpenGraph[Measurement]:
    graph: Graph = nx.Graph(
        [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
        ]
    )
    output_nodes = tuple(node for node in range(4) if node not in measurements)
    return OpenGraph(graph, input_nodes=(1, 2, 3), output_nodes=output_nodes, measurements=measurements)


def opengraph_lemma_2_32(measurements: Mapping[Node, Measurement]) -> OpenGraph[Measurement]:
    graph: Graph = nx.Graph(
        [
            (0, 1),
            (0, 2),
            (1, 2),
            (0, 3),
            (1, 3),
            (0, 4),
            (3, 4),
            (1, 5),
            (3, 5),
            (0, 6),
            (2, 6),
            (5, 6),
            (1, 7),
            (2, 7),
            (4, 7),
        ]
    )
    output_nodes = tuple(node for node in range(8) if node not in measurements)
    return OpenGraph(graph, input_nodes=(4, 5, 6, 7), output_nodes=output_nodes, measurements=measurements)


@pytest.mark.parametrize("measured_set", [set(), {1}, {2}])
def test_local_complement(fx_rng: Generator, measured_set: AbstractSet[int]) -> None:
    og = opengraph_lemma_2_31({node: Measurement.XY(0.25) for node in measured_set})
    pattern = og.to_pattern()
    standardized_pattern = StandardizedPattern.from_pattern(pattern)
    cut = PauliPushingCut.from_standardizedpattern(standardized_pattern)
    remove_pauli_measurements = _RemovePauliMeasurements(cut)
    remove_pauli_measurements.local_complement(0)
    standardized_pattern2 = remove_pauli_measurements.to_standardizedpattern()
    og2 = standardized_pattern2.to_opengraph()
    expected_graph: Graph = nx.Graph(
        [
            (0, 1),
            (0, 2),
            (0, 3),
            (2, 3),
        ]
    )
    assert nx.utils.graphs_equal(og2.graph, expected_graph)
    pattern2 = standardized_pattern2.to_pattern()
    assert pattern2.to_gflow()
    check_pattern_equivalence(pattern, pattern2, rng=fx_rng)


@pytest.mark.parametrize("measured_set", [set(), {4}, {4, 5}, {4, 5, 6}, {4, 5, 7}, {0}, {1}, {2}, {3}, {0, 2}])
def test_pivot_edge(fx_rng: Generator, measured_set: AbstractSet[int]) -> None:
    og = opengraph_lemma_2_32({node: Measurement.XY(0.25) for node in measured_set})
    pattern = og.to_pattern()
    standardized_pattern = StandardizedPattern.from_pattern(pattern)
    cut = PauliPushingCut.from_standardizedpattern(standardized_pattern)
    remove_pauli_measurements = _RemovePauliMeasurements(cut)
    remove_pauli_measurements.pivot_edge(0, 1)
    standardized_pattern2 = remove_pauli_measurements.to_standardizedpattern()
    og2 = standardized_pattern2.to_opengraph()
    expected_graph: Graph = nx.Graph(
        [
            (0, 1),
            (0, 2),
            (1, 2),
            (0, 3),
            (1, 3),
            (0, 4),
            (2, 4),
            (1, 5),
            (2, 5),
            (4, 5),
            (0, 6),
            (3, 6),
            (1, 7),
            (3, 7),
            (6, 7),
        ]
    )
    assert nx.utils.graphs_equal(og2.graph, expected_graph)
    assert og2.output_nodes == tuple(0 if node == 1 else 1 if node == 0 else node for node in og.output_nodes)
    pattern2 = standardized_pattern2.to_pattern()
    assert pattern2.to_gflow()
    check_pattern_equivalence(pattern, pattern2, rng=fx_rng)


@pytest.mark.parametrize("node", [0, 1, 2, 3])
@pytest.mark.parametrize("sign", Sign)
def test_remove_z(fx_rng: Generator, node: Node, sign: Sign) -> None:
    og = opengraph_lemma_2_32({node: PauliMeasurement(Axis.Z, sign)})
    pattern = og.to_pattern()
    standardized_pattern = StandardizedPattern.from_pattern(pattern)
    cut = PauliPushingCut.from_standardizedpattern(standardized_pattern)
    remove_pauli_measurements = _RemovePauliMeasurements(cut)
    remove_pauli_measurements.remove_z(node, sign)
    standardized_pattern2 = remove_pauli_measurements.to_standardizedpattern()
    pattern2 = standardized_pattern2.to_pattern()
    check_pattern_equivalence(pattern, pattern2, rng=fx_rng)


@pytest.mark.parametrize("node", [0, 1, 2, 3])
@pytest.mark.parametrize("sign", Sign)
def test_remove_y(fx_rng: Generator, node: Node, sign: Sign) -> None:
    og = opengraph_lemma_2_32({node: PauliMeasurement(Axis.Y, sign)})
    pattern = og.to_pattern()
    standardized_pattern = StandardizedPattern.from_pattern(pattern)
    cut = PauliPushingCut.from_standardizedpattern(standardized_pattern)
    remove_pauli_measurements = _RemovePauliMeasurements(cut)
    remove_pauli_measurements.remove_y(node, sign)
    standardized_pattern2 = remove_pauli_measurements.to_standardizedpattern()
    pattern2 = standardized_pattern2.to_pattern()
    check_pattern_equivalence(pattern, pattern2, rng=fx_rng)


@pytest.mark.parametrize("sign", Sign)
def test_remove_x_with_internal_neighbor(fx_rng: Generator, sign: Sign) -> None:
    og = opengraph_lemma_2_32({0: PauliMeasurement(Axis.X, sign)})
    pattern = og.to_pattern()
    standardized_pattern = StandardizedPattern.from_pattern(pattern)
    cut = PauliPushingCut.from_standardizedpattern(standardized_pattern)
    remove_pauli_measurements = _RemovePauliMeasurements(cut)
    remove_pauli_measurements.remove_x_with_internal_neighbor(0, 1, sign)
    standardized_pattern2 = remove_pauli_measurements.to_standardizedpattern()
    pattern2 = standardized_pattern2.to_pattern()
    check_pattern_equivalence(pattern, pattern2, rng=fx_rng)


def all_bloch_measurement_or_input_node(input_nodes: Iterable[Node], measurement_commands: Iterable[Command.M]) -> bool:
    input_node_set = set(input_nodes)
    return all(
        isinstance(cmd_m.measurement, BlochMeasurement) or cmd_m.node in input_node_set
        for cmd_m in measurement_commands
    )


def check_pattern(pattern: Pattern, rng: Generator) -> None:
    standardized_pattern = StandardizedPattern.from_pattern(pattern)
    cut = PauliPushingCut.from_standardizedpattern(standardized_pattern)
    standardized_pattern2 = remove_pauli_measurements(cut)

    assert all_bloch_measurement_or_input_node(standardized_pattern2.input_nodes, standardized_pattern2.m_list)

    # Check that the pattern has a gflow
    standardized_pattern2.to_xzcorrections().to_bloch().to_gflow()

    pattern2 = standardized_pattern2.to_pattern()
    check_pattern_equivalence(pattern, pattern2, rng=rng)


def check_pattern_equivalence(pattern: Pattern, pattern2: Pattern, rng: Generator) -> None:
    pattern.minimize_space()
    pattern2.minimize_space()
    for _ in range(4):
        input_state = rand_state_vector(len(pattern.input_nodes), rng=rng)
        state = pattern.simulate_pattern(input_state=input_state, rng=rng)
        state2 = pattern2.simulate_pattern(input_state=input_state, rng=rng)
        assert state.isclose(state2)


def test_ccx(fx_rng: Generator) -> None:
    circuit = Circuit(3)
    circuit.ccx(0, 1, 2)
    check_pattern(circuit.transpile().pattern, fx_rng)


@pytest.mark.parametrize("jumps", range(1, 11))
def test_random_circuit(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 4
    depth = 4
    circuit = rand_circuit(nqubits, depth, rng)
    check_pattern(circuit.transpile().pattern, rng)


def test_step_4() -> None:
    graph: Graph = nx.Graph([(0, 1), (1, 2)])
    measurements = {0: Measurement.XY(0.25), 1: Measurement.X}
    og = OpenGraph(graph, input_nodes=(0,), output_nodes=(2,), measurements=measurements)
    pattern = og.to_pattern()
    standardized_pattern = StandardizedPattern.from_pattern(pattern)
    cut = PauliPushingCut.from_standardizedpattern(standardized_pattern)
    standardized_pattern2 = remove_pauli_measurements(cut)
    assert len(standardized_pattern2.m_list) == 1


def test_step_4_no_flow() -> None:
    # This example tests the case of a pattern that contains a
    # non-input X-measured node 1 which is connected to an output node
    # 0, where the node 0 is also an input.  In this situation Lemma
    # 4.11 cannot be applied; this exercices the filtering implemented
    # in the `try_pivot_x_with_output_node` method.
    pattern = Pattern(input_nodes=(0,), output_nodes=(0,), cmds=[Command.N(1), Command.E((0, 1)), Command.M(1)])
    standardized_pattern = StandardizedPattern.from_pattern(pattern)
    cut = PauliPushingCut.from_standardizedpattern(standardized_pattern)
    standardized_pattern2 = remove_pauli_measurements(cut)
    assert len(standardized_pattern2.m_list) == 1


def test_cliffords_in_original_pattern(fx_rng: Generator) -> None:
    circuit = Circuit(2)
    circuit.cnot(0, 1)
    pattern = circuit.transpile().pattern
    u, v = pattern.output_nodes
    pattern.add(Command.C(u, Clifford.S))
    pattern.add(Command.C(v, Clifford.SDG))
    check_pattern(pattern, fx_rng)


def test_pattern_remove_pauli_measurements() -> None:
    circuit = Circuit(2)
    circuit.cnot(0, 1)
    pattern = circuit.transpile().pattern
    pattern = pattern.infer_pauli_measurements()
    pattern2 = pattern.remove_pauli_measurements(copy=True)
    assert all_bloch_measurement_or_input_node(
        pattern2.input_nodes, (cmd for cmd in pattern2 if isinstance(cmd, Command.M))
    )
    assert not pattern2.is_standard()
    pattern3 = pattern.remove_pauli_measurements(copy=True, standardize=True)
    assert all_bloch_measurement_or_input_node(
        pattern3.input_nodes, (cmd for cmd in pattern3 if isinstance(cmd, Command.M))
    )
    assert pattern3.is_standard()
    assert not all_bloch_measurement_or_input_node(
        pattern.input_nodes, (cmd for cmd in pattern if isinstance(cmd, Command.M))
    )
    pattern.remove_pauli_measurements()
    assert all_bloch_measurement_or_input_node(
        pattern.input_nodes, (cmd for cmd in pattern if isinstance(cmd, Command.M))
    )


def test_pattern_remove_pauli_measurements_output_nodes() -> None:
    og = OpenGraph(
        graph=nx.Graph([(1, 2)]),
        input_nodes=[],
        output_nodes=[2],
        measurements={
            1: Measurement.X,
        },
    )
    pattern = og.to_pattern()
    pattern.remove_pauli_measurements()
    pattern.simulate_pattern()


def test_try_pivot_x_with_output_node_after_pivot() -> None:
    # This test checks that `try_pivot_x_with_output_node` applies
    # `pivot_edge` using `new_node` rather than the original
    # `node`.
    #
    # In practice this situation is unlikely to arise: for `node != new_node`
    # to occur, a pivot must have already been applied to `node`.  Yet,
    # after such a pivot we would need `new_node` to be measured in X, which
    # implies that `node` was originally measured in Z.  The removal strategy
    # would then delete `node` before the pivot could take place.
    #
    # Consequently, this test guarantees that `try_pivot_x_with_output_node`
    # works correctly regardless of the removal strategy and maintains the
    # intended invariant, even though the earlier bug (pivoting with the
    # original node) was not observable through the public API.
    pattern = Pattern(
        cmds=[
            Command.N(0),
            Command.N(1),
            Command.N(2),
            Command.E((0, 1)),
            Command.E((0, 2)),
            Command.M(0),
            Command.M(1, Measurement.Z),
        ]
    )
    standardized_pattern = StandardizedPattern.from_pattern(pattern)
    cut = PauliPushingCut.from_standardizedpattern(standardized_pattern)
    process = _RemovePauliMeasurements(cut)
    process.remove_x_with_internal_neighbor(0, 1, Sign.PLUS)
    # Fail if pivot is applied to the original node
    process.try_pivot_x_with_output_node()


def test_isolated_nodes_non_pauli() -> None:
    pattern = Pattern(cmds=[Command.N(0), Command.N(1), Command.M(0), Command.M(1, Measurement.XY(ANGLE_PI / 4))])
    with pytest.warns(UserWarning, match="Non-Pauli measurement on an isolated node was removed."):
        pattern.remove_pauli_measurements()
    assert list(pattern) == []
