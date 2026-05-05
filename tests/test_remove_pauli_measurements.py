from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import pytest
from numpy.random import Generator

from graphix import Axis, BlochMeasurement, Circuit, Measurement, OpenGraph, PauliMeasurement, Sign, StandardizedPattern
from graphix.random_objects import rand_circuit, rand_state_vector
from graphix.remove_pauli_measurements import PauliPushingCut, _RemovePauliMeasurements, remove_pauli_measurements

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet

    from numpy.random import PCG64

    from graphix.command import Node
    from graphix.pattern import Pattern
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
    cut = PauliPushingCut.from_standardized_pattern(standardized_pattern)
    remove_pauli_measurements = _RemovePauliMeasurements(cut)
    remove_pauli_measurements.local_complement(0)
    standardized_pattern2 = remove_pauli_measurements.to_standardized_pattern()
    og2 = standardized_pattern2.extract_opengraph()
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
    assert pattern2.extract_gflow()
    check_pattern_equivalence(pattern, pattern2, rng=fx_rng)


@pytest.mark.parametrize("measured_set", [set(), {4}, {4, 5}, {4, 5, 6}, {4, 5, 7}, {0}, {1}, {2}, {3}, {0, 2}])
def test_pivot_vertices(fx_rng: Generator, measured_set: AbstractSet[int]) -> None:
    og = opengraph_lemma_2_32({node: Measurement.XY(0.25) for node in measured_set})
    pattern = og.to_pattern()
    standardized_pattern = StandardizedPattern.from_pattern(pattern)
    cut = PauliPushingCut.from_standardized_pattern(standardized_pattern)
    remove_pauli_measurements = _RemovePauliMeasurements(cut)
    remove_pauli_measurements.pivot_vertices(0, 1)
    standardized_pattern2 = remove_pauli_measurements.to_standardized_pattern()
    og2 = standardized_pattern2.extract_opengraph()
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
    assert pattern2.extract_gflow()
    check_pattern_equivalence(pattern, pattern2, rng=fx_rng)


@pytest.mark.parametrize("node", [0, 1, 2, 3])
@pytest.mark.parametrize("sign", Sign)
def test_remove_z(fx_rng: Generator, node: Node, sign: Sign) -> None:
    og = opengraph_lemma_2_32({node: PauliMeasurement(Axis.Z, sign)})
    pattern = og.to_pattern()
    standardized_pattern = StandardizedPattern.from_pattern(pattern)
    cut = PauliPushingCut.from_standardized_pattern(standardized_pattern)
    remove_pauli_measurements = _RemovePauliMeasurements(cut)
    remove_pauli_measurements.remove_z(node, sign)
    standardized_pattern2 = remove_pauli_measurements.to_standardized_pattern()
    pattern2 = standardized_pattern2.to_pattern()
    check_pattern_equivalence(pattern, pattern2, rng=fx_rng)


@pytest.mark.parametrize("node", [0, 1, 2, 3])
@pytest.mark.parametrize("sign", Sign)
def test_remove_y(fx_rng: Generator, node: Node, sign: Sign) -> None:
    og = opengraph_lemma_2_32({node: PauliMeasurement(Axis.Y, sign)})
    pattern = og.to_pattern()
    standardized_pattern = StandardizedPattern.from_pattern(pattern)
    cut = PauliPushingCut.from_standardized_pattern(standardized_pattern)
    remove_pauli_measurements = _RemovePauliMeasurements(cut)
    remove_pauli_measurements.remove_y(node, sign)
    standardized_pattern2 = remove_pauli_measurements.to_standardized_pattern()
    pattern2 = standardized_pattern2.to_pattern()
    check_pattern_equivalence(pattern, pattern2, rng=fx_rng)


@pytest.mark.parametrize("sign", Sign)
def test_remove_x_with_non_input_neighbor(fx_rng: Generator, sign: Sign) -> None:
    og = opengraph_lemma_2_32({0: PauliMeasurement(Axis.X, sign)})
    pattern = og.to_pattern()
    standardized_pattern = StandardizedPattern.from_pattern(pattern)
    cut = PauliPushingCut.from_standardized_pattern(standardized_pattern)
    remove_pauli_measurements = _RemovePauliMeasurements(cut)
    remove_pauli_measurements.remove_x_with_non_input_neighbor(0, 1, sign)
    standardized_pattern2 = remove_pauli_measurements.to_standardized_pattern()
    pattern2 = standardized_pattern2.to_pattern()
    check_pattern_equivalence(pattern, pattern2, rng=fx_rng)


def check_circuit(circuit: Circuit, rng: Generator) -> None:
    pattern = circuit.transpile().pattern
    standardized_pattern = StandardizedPattern.from_pattern(pattern)
    standardized_pattern2 = remove_pauli_measurements(standardized_pattern)

    input_node_set = set(standardized_pattern2.input_nodes)
    assert all(
        isinstance(cmd_m.measurement, BlochMeasurement) or cmd_m.node in input_node_set
        for cmd_m in standardized_pattern2.m_list
    )

    # Check that the pattern has a gflow
    standardized_pattern2.to_bloch().extract_gflow()

    pattern2 = standardized_pattern2.to_pattern()
    check_pattern_equivalence(pattern, pattern2, rng=rng)


def test_ccx(fx_rng: Generator) -> None:
    circuit = Circuit(3)
    circuit.ccx(0, 1, 2)
    check_circuit(circuit, fx_rng)


@pytest.mark.parametrize("jumps", range(1, 11))
def test_random_circuit(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 4
    depth = 4
    circuit = rand_circuit(nqubits, depth, rng)
    check_circuit(circuit, rng)


def check_pattern_equivalence(pattern: Pattern, pattern2: Pattern, rng: Generator) -> None:
    pattern.minimize_space()
    pattern2.minimize_space()
    for _ in range(4):
        input_state = rand_state_vector(len(pattern.input_nodes), rng=rng)
        state = pattern.simulate_pattern(input_state=input_state, rng=rng)
        state2 = pattern2.simulate_pattern(input_state=input_state, rng=rng)
        assert state.isclose(state2)
