"""Unit tests for the class `:class: graphix.opengraph.OpenGraph`.

This module tests the full conversion Open Graph -> Flow -> XZ-corrections -> Pattern for all three classes of flow.
Output correctness is verified by checking if the resulting pattern is deterministic (when the flow exists).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import networkx as nx
import numpy as np
import pytest

from graphix.command import E
from graphix.fundamentals import Plane
from graphix.measurements import Measurement
from graphix.opengraph import OpenGraph
from graphix.pattern import Pattern
from graphix.random_objects import rand_circuit
from graphix.states import PlanarState

if TYPE_CHECKING:
    from numpy.random import Generator


class OpenGraphFlowTestCase(NamedTuple):
    og: OpenGraph[Measurement]
    has_cflow: bool
    has_gflow: bool
    has_pflow: bool


def _og_0() -> OpenGraphFlowTestCase:
    """Generate open graph.

    Structure:

    [(0)]-[(1)]
    """
    meas: dict[int, Measurement] = {}
    og = OpenGraph(
        graph=nx.Graph([(0, 1)]),
        input_nodes=[0, 1],
        output_nodes=[0, 1],
        measurements=meas,
    )
    return OpenGraphFlowTestCase(og, has_cflow=True, has_gflow=True, has_pflow=True)


def _og_1() -> OpenGraphFlowTestCase:
    """Generate open graph.

    Structure:

    [0]-1-20-30-4-(5)
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 1), (1, 20), (20, 30), (30, 4), (4, 5)]),
        input_nodes=[0],
        output_nodes=[5],
        measurements={
            0: Measurement(0.1, Plane.XY),
            1: Measurement(0.2, Plane.XY),
            20: Measurement(0.3, Plane.XY),
            30: Measurement(0.4, Plane.XY),
            4: Measurement(0.5, Plane.XY),
        },
    )
    return OpenGraphFlowTestCase(og, has_cflow=True, has_gflow=True, has_pflow=True)


def _og_2() -> OpenGraphFlowTestCase:
    """Generate open graph.

    Structure:

    [1]-3-(5)
        |
    [2]-4-(6)
    """
    og = OpenGraph(
        graph=nx.Graph([(1, 3), (2, 4), (3, 4), (3, 5), (4, 6)]),
        input_nodes=[1, 2],
        output_nodes=[5, 6],
        measurements={
            1: Measurement(0.1, Plane.XY),
            2: Measurement(0.2, Plane.XY),
            3: Measurement(0.3, Plane.XY),
            4: Measurement(0.4, Plane.XY),
        },
    )
    return OpenGraphFlowTestCase(og, has_cflow=True, has_gflow=True, has_pflow=True)


def _og_3() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

    [1]-(4)
      \ /
       X____
      /     |
    [2]-(5) |
      \ /   |
       X    |
      / \  /
    [3]-(6)
    """
    og = OpenGraph(
        graph=nx.Graph([(1, 4), (1, 6), (2, 4), (2, 5), (2, 6), (3, 5), (3, 6)]),
        input_nodes=[1, 2, 3],
        output_nodes=[4, 5, 6],
        measurements={
            1: Measurement(0.1, Plane.XY),
            2: Measurement(0.2, Plane.XY),
            3: Measurement(0.3, Plane.XY),
        },
    )
    return OpenGraphFlowTestCase(og, has_cflow=False, has_gflow=True, has_pflow=True)


def _og_4() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

       [0]-[1]
       /|   |
     (4)|   |
       \|   |
        2--(5)-3
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 1), (0, 2), (0, 4), (1, 5), (2, 4), (2, 5), (3, 5)]),
        input_nodes=[0, 1],
        output_nodes=[4, 5],
        measurements={
            0: Measurement(0.1, Plane.XY),
            1: Measurement(0.1, Plane.XY),
            2: Measurement(0.2, Plane.XZ),
            3: Measurement(0.3, Plane.YZ),
        },
    )
    return OpenGraphFlowTestCase(og, has_cflow=False, has_gflow=True, has_pflow=True)


def _og_5() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

    [1]-(3)
      \ /
       X
      / \
    [2]-(4)
    """
    og = OpenGraph(
        graph=nx.Graph([(1, 3), (1, 4), (2, 3), (2, 4)]),
        input_nodes=[1, 2],
        output_nodes=[3, 4],
        measurements={
            1: Measurement(0.1, Plane.XY),
            2: Measurement(0.1, Plane.XY),
        },
    )
    return OpenGraphFlowTestCase(og, has_cflow=False, has_gflow=False, has_pflow=False)


def _og_6() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

    [0]
     |
     1-2-3
     |
    (4)
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 1), (1, 2), (1, 4), (2, 3)]),
        input_nodes=[0],
        output_nodes=[4],
        measurements={
            0: Measurement(0.1, Plane.XY),  # XY
            1: Measurement(0, Plane.XY),  # X
            2: Measurement(0.1, Plane.XY),  # XY
            3: Measurement(0, Plane.XY),  # X
        },
    )
    return OpenGraphFlowTestCase(og, has_cflow=False, has_gflow=False, has_pflow=True)


def _og_7() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

    [0]-1-(2)
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 1), (1, 2)]),
        input_nodes=[0],
        output_nodes=[2],
        measurements={
            0: Measurement(0.1, Plane.XY),  # XY
            1: Measurement(0.5, Plane.YZ),  # Y
        },
    )
    return OpenGraphFlowTestCase(og, has_cflow=False, has_gflow=False, has_pflow=True)


def _og_8() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

    [0]-2-4-(6)
        | |
    [1]-3-5-(7)
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5), (4, 6), (5, 7)]),
        input_nodes=[1, 0],
        output_nodes=[6, 7],
        measurements={
            0: Measurement(0.1, Plane.XY),  # XY
            1: Measurement(0.1, Plane.XZ),  # XZ
            2: Measurement(0.5, Plane.XZ),  # X
            3: Measurement(0.5, Plane.YZ),  # Y
            4: Measurement(0.5, Plane.YZ),  # Y
            5: Measurement(0.1, Plane.YZ),  # YZ
        },
    )
    return OpenGraphFlowTestCase(og, has_cflow=False, has_gflow=False, has_pflow=False)


def _og_9() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

    [0]-2-4-(6)
        | |
    [1]-3-5-(7)
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5), (4, 6), (5, 7)]),
        input_nodes=[0, 1],
        output_nodes=[6, 7],
        measurements={
            0: Measurement(0.1, Plane.XY),  # XY
            1: Measurement(0.1, Plane.XY),  # XY
            2: Measurement(0.0, Plane.XY),  # X
            3: Measurement(0.1, Plane.XY),  # XY
            4: Measurement(0.0, Plane.XY),  # X
            5: Measurement(0.5, Plane.XY),  # Y
        },
    )
    return OpenGraphFlowTestCase(og, has_cflow=True, has_gflow=True, has_pflow=True)


def _og_10() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

          3-(5)
         /| \
    [0]-2-4-(6)
        | | /|
        | 1  |
        |____|

    Notes
    -----
    Example from Fig. 1 in Mitosek and Backens, 2024 (arXiv:2410.23439).
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 2), (2, 4), (3, 4), (4, 6), (1, 4), (1, 6), (2, 3), (3, 5), (2, 6), (3, 6)]),
        input_nodes=[0],
        output_nodes=[5, 6],
        measurements={
            0: Measurement(0.1, Plane.XY),  # XY
            1: Measurement(0.1, Plane.XZ),  # XZ
            2: Measurement(0.5, Plane.YZ),  # Y
            3: Measurement(0.1, Plane.XY),  # XY
            4: Measurement(0, Plane.XZ),  # Z
        },
    )
    return OpenGraphFlowTestCase(og, has_cflow=False, has_gflow=False, has_pflow=True)


def _og_11() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

    [0]-2--3--4-7-(8)
        |  |  |
       (6)[1](5)
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 2), (1, 3), (2, 3), (2, 6), (3, 4), (4, 7), (4, 5), (7, 8)]),
        input_nodes=[0, 1],
        output_nodes=[5, 6, 8],
        measurements={
            0: Measurement(0.1, Plane.XY),
            1: Measurement(0.1, Plane.XY),
            2: Measurement(0.0, Plane.XY),
            3: Measurement(0, Plane.XY),
            4: Measurement(0.5, Plane.XY),
            7: Measurement(0, Plane.XY),
        },
    )
    return OpenGraphFlowTestCase(og, has_cflow=True, has_gflow=True, has_pflow=True)


def _og_12() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

    [0]-2-(3)-(4)
        |
      [(1)]
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 2), (1, 2), (2, 3), (3, 4)]),
        input_nodes=[0, 1],
        output_nodes=[1, 3, 4],
        measurements={0: Measurement(0.1, Plane.XY), 2: Measurement(0.5, Plane.YZ)},
    )
    return OpenGraphFlowTestCase(og, has_cflow=False, has_gflow=False, has_pflow=True)


def _og_13() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

          0-[1]
          |  |
    5-(2)-3--4-(7)
          |
         (6)
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 1), (0, 3), (1, 4), (3, 4), (2, 3), (2, 5), (3, 6), (4, 7)]),
        input_nodes=[1],
        output_nodes=[6, 2, 7],
        measurements={
            0: Measurement(0.1, Plane.XZ),
            1: Measurement(0.1, Plane.XY),
            3: Measurement(0, Plane.XY),
            4: Measurement(0.1, Plane.XY),
            5: Measurement(0.1, Plane.YZ),
        },
    )
    return OpenGraphFlowTestCase(og, has_cflow=False, has_gflow=True, has_pflow=True)


def _og_14() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

    0-(1)  (4)-6
    |  |
    2-(3)
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 1), (0, 2), (2, 3), (1, 3), (4, 6)]),
        input_nodes=[],
        output_nodes=[1, 3, 4],
        measurements={0: Measurement(0.5, Plane.XZ), 2: Measurement(0, Plane.YZ), 6: Measurement(0.2, Plane.XY)},
    )
    return OpenGraphFlowTestCase(og, has_cflow=False, has_gflow=True, has_pflow=True)


def _og_15() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

    [1]--0
     |   |
     4---3-(2)
     |   |  |
    (7)-(6)-5
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 1), (0, 3), (1, 4), (3, 4), (2, 3), (2, 5), (3, 6), (4, 7), (5, 6), (6, 7)]),
        input_nodes=[1],
        output_nodes=[6, 2, 7],
        measurements={
            0: Measurement(0.1, Plane.XZ),
            1: Measurement(0.1, Plane.XY),
            3: Measurement(0, Plane.XY),
            4: Measurement(0.1, Plane.XY),
            5: Measurement(0.1, Plane.XY),
        },
    )
    return OpenGraphFlowTestCase(og, has_cflow=False, has_gflow=False, has_pflow=False)


def _og_16() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

    [0]-(1)  (4)-6
     |   |
     2--(3)
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 1), (0, 2), (2, 3), (1, 3), (4, 6)]),
        input_nodes=[0],
        output_nodes=[1, 3, 4],
        measurements={0: Measurement(0.1, Plane.XZ), 2: Measurement(0, Plane.YZ), 6: Measurement(0.2, Plane.XY)},
    )
    return OpenGraphFlowTestCase(og, has_cflow=False, has_gflow=False, has_pflow=False)


def _og_17() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

          0-[1]     [8]
          |  |
    5-(2)-3--4-(7)
          |
         (6)

    Notes
    -----
    Graph is constructed by adding a disconnected input to OG 13.
    """
    graph: nx.Graph[int] = nx.Graph([(0, 1), (0, 3), (1, 4), (3, 4), (2, 3), (2, 5), (3, 6), (4, 7)])
    graph.add_node(8)
    og = OpenGraph(
        graph=graph,
        input_nodes=[1, 8],
        output_nodes=[6, 2, 7],
        measurements={
            0: Measurement(0.1, Plane.XZ),
            1: Measurement(0.1, Plane.XY),
            3: Measurement(0, Plane.XY),
            4: Measurement(0.1, Plane.XY),
            5: Measurement(0.1, Plane.YZ),
            8: Measurement(0.1, Plane.XY),
        },
    )
    return OpenGraphFlowTestCase(og, has_cflow=False, has_gflow=False, has_pflow=False)


def _og_18() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

    [0]-2--3--4-7-(8)
        |  |  |
       (6)[1](5)
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 2), (1, 3), (2, 3), (2, 6), (3, 4), (4, 7), (4, 5), (7, 8)]),
        input_nodes=[0, 1],
        output_nodes=[5, 6, 8],
        measurements={
            0: Measurement(0, Plane.XY),
            1: Measurement(0, Plane.XY),
            2: Measurement(0, Plane.XZ),
            3: Measurement(0, Plane.XY),
            4: Measurement(0.5, Plane.XY),
            7: Measurement(0, Plane.YZ),
        },
    )
    return OpenGraphFlowTestCase(og, has_cflow=False, has_gflow=False, has_pflow=False)


def _og_19() -> OpenGraphFlowTestCase:
    r"""Generate open graph.

    Structure:

    0-2--3--4-7-(8)
      |  |  |
     (6) 1 (5)

    Notes
    -----
    Even though any node is measured in the XY plane, OG has Pauli flow because none of them are inputs.
    """
    og = OpenGraph(
        graph=nx.Graph([(0, 2), (1, 3), (2, 3), (2, 6), (3, 4), (4, 7), (4, 5), (7, 8)]),
        input_nodes=[],
        output_nodes=[5, 6, 8],
        measurements={
            0: Measurement(0, Plane.XZ),
            1: Measurement(0, Plane.XZ),
            2: Measurement(0, Plane.XZ),
            3: Measurement(0, Plane.XZ),
            4: Measurement(0, Plane.XZ),
            7: Measurement(0, Plane.XZ),
        },
    )
    return OpenGraphFlowTestCase(og, has_cflow=False, has_gflow=False, has_pflow=True)


def prepare_test_og_flow() -> list[OpenGraphFlowTestCase]:
    n_og_samples = 20
    test_cases: list[OpenGraphFlowTestCase] = [globals()[f"_og_{i}"]() for i in range(n_og_samples)]

    return test_cases


def check_determinism(pattern: Pattern, fx_rng: Generator, n_shots: int = 3) -> bool:
    """Verify if the input pattern is deterministic."""
    results = []

    for plane in {Plane.XY, Plane.XZ, Plane.YZ}:
        alpha = 2 * np.pi * fx_rng.random()
        state_ref = pattern.simulate_pattern(input_state=PlanarState(plane, alpha))

        for _ in range(n_shots):
            state = pattern.simulate_pattern(input_state=PlanarState(plane, alpha))
            results.append(np.abs(np.dot(state.flatten().conjugate(), state_ref.flatten())))

    avg = sum(results) / (n_shots * 3)

    return bool(avg == pytest.approx(1))


class TestOpenGraph:
    def test_odd_neighbors(self) -> None:
        graph: nx.Graph[int] = nx.Graph([(0, 1), (0, 2), (1, 3), (1, 2), (2, 3), (1, 4)])
        og = OpenGraph(graph=graph, input_nodes=[0], output_nodes=[1, 2, 3, 4], measurements={0: Plane.XY})

        assert og.odd_neighbors([0]) == {1, 2}
        assert og.odd_neighbors([0, 1]) == {0, 1, 3, 4}
        assert og.odd_neighbors([1, 2, 3]) == {4}
        assert og.odd_neighbors([]) == set()

    def test_neighbors(self) -> None:
        graph: nx.Graph[int] = nx.Graph([(0, 1), (0, 2), (1, 3), (1, 2), (2, 3), (1, 4)])
        og = OpenGraph(graph=graph, input_nodes=[0], output_nodes=[1, 2, 3, 4], measurements={0: Plane.XY})

        assert og.neighbors([4]) == {1}
        assert og.neighbors([0, 1]) == {0, 1, 2, 3, 4}
        assert og.neighbors([1, 2, 3]) == {0, 1, 2, 3, 4}
        assert og.neighbors([]) == set()

    @pytest.mark.parametrize("test_case", prepare_test_og_flow())
    def test_cflow(self, test_case: OpenGraphFlowTestCase, fx_rng: Generator) -> None:
        og = test_case.og

        cflow = og.find_causal_flow()

        if test_case.has_cflow:
            assert cflow is not None
            pattern = cflow.to_corrections().to_pattern()
            assert check_determinism(pattern, fx_rng)
        else:
            assert cflow is None

    @pytest.mark.parametrize("test_case", prepare_test_og_flow())
    def test_gflow(self, test_case: OpenGraphFlowTestCase, fx_rng: Generator) -> None:
        og = test_case.og

        gflow = og.find_gflow()

        if test_case.has_gflow:
            assert gflow is not None
            pattern = gflow.to_corrections().to_pattern()
            assert check_determinism(pattern, fx_rng)
        else:
            assert gflow is None

    @pytest.mark.parametrize("test_case", prepare_test_og_flow())
    def test_pflow(self, test_case: OpenGraphFlowTestCase, fx_rng: Generator) -> None:
        og = test_case.og

        pflow = og.find_pauli_flow()

        if test_case.has_pflow:
            assert pflow is not None
            pattern = pflow.to_corrections().to_pattern()
            assert check_determinism(pattern, fx_rng)
        else:
            assert pflow is None

    def test_double_entanglement(self) -> None:
        pattern = Pattern(input_nodes=[0, 1], cmds=[E((0, 1)), E((0, 1))])
        pattern2 = OpenGraph.from_pattern(pattern).to_pattern()
        state = pattern.simulate_pattern()
        assert pattern2 is not None
        state2 = pattern2.simulate_pattern()
        assert np.abs(np.dot(state.flatten().conjugate(), state2.flatten())) == pytest.approx(1)

    def test_from_to_pattern(self, fx_rng: Generator) -> None:
        n_qubits = 2
        depth = 2
        circuit = rand_circuit(n_qubits, depth, fx_rng)
        pattern_ref = circuit.transpile().pattern
        pattern = OpenGraph.from_pattern(pattern_ref).to_pattern()
        assert pattern is not None

        results = []

        for plane in {Plane.XY, Plane.XZ, Plane.YZ}:
            alpha = 2 * np.pi * fx_rng.random()
            state_ref = pattern_ref.simulate_pattern(input_state=PlanarState(plane, alpha))
            state = pattern.simulate_pattern(input_state=PlanarState(plane, alpha))
            results.append(np.abs(np.dot(state.flatten().conjugate(), state_ref.flatten())))

        avg = sum(results) / 3
        assert avg == pytest.approx(1)


# TODO: Add test `OpenGraph.is_close`
# TODO: rewrite as parametric tests

# Tests composition of two graphs


# Parallel composition
def test_compose_1() -> None:
    # Graph 1
    # [1] -- (2)
    #
    # Graph 2 = Graph 1
    #
    # Mapping: 1 -> 100, 2 -> 200
    #
    # Expected graph
    #  [1]  --  (2)
    #
    # [100] -- (200)

    g: nx.Graph[int]
    g = nx.Graph([(1, 2)])
    inputs = [1]
    outputs = [2]
    meas = {i: Measurement(0, Plane.XY) for i in g.nodes - set(outputs)}
    og_1 = OpenGraph(g, inputs, outputs, meas)

    mapping = {1: 100, 2: 200}

    og, mapping_complete = og_1.compose(og_1, mapping)

    expected_graph: nx.Graph[int]
    expected_graph = nx.Graph([(1, 2), (100, 200)])
    assert nx.is_isomorphic(og.graph, expected_graph)
    assert og.input_nodes == [1, 100]
    assert og.output_nodes == [2, 200]

    outputs_c = {i for i in og.graph.nodes if i not in og.output_nodes}
    assert og.measurements.keys() == outputs_c
    assert mapping.keys() <= mapping_complete.keys()
    assert set(mapping.values()) <= set(mapping_complete.values())


# Series composition
def test_compose_2() -> None:
    # Graph 1
    # [0] -- 17 -- (23)
    #        |
    # [3] -- 4  -- (13)
    #
    # Graph 2
    # [6] -- 17 -- (1)
    #  |     |
    # [7] -- 4  -- (2)
    #
    # Mapping: 6 -> 23, 7 -> 13, 1 -> 100, 2 -> 200
    #
    # Expected graph
    # [0] -- 17 -- 23 -- o -- (100)
    #        |     |     |
    # [3] -- 4  -- 13 -- o -- (200)

    g: nx.Graph[int]
    g = nx.Graph([(0, 17), (17, 23), (17, 4), (3, 4), (4, 13)])
    inputs = [0, 3]
    outputs = [13, 23]
    meas = {i: Measurement(0, Plane.XY) for i in g.nodes - set(outputs)}
    og_1 = OpenGraph(g, inputs, outputs, meas)

    g = nx.Graph([(6, 7), (6, 17), (17, 1), (7, 4), (17, 4), (4, 2)])
    inputs = [6, 7]
    outputs = [1, 2]
    meas = {i: Measurement(0, Plane.XY) for i in g.nodes - set(outputs)}
    og_2 = OpenGraph(g, inputs, outputs, meas)

    mapping = {6: 23, 7: 13, 1: 100, 2: 200}

    og, mapping_complete = og_1.compose(og_2, mapping)

    expected_graph: nx.Graph[int]
    expected_graph = nx.Graph(
        [(0, 17), (17, 23), (17, 4), (3, 4), (4, 13), (23, 13), (23, 1), (13, 2), (1, 2), (1, 100), (2, 200)]
    )
    assert nx.is_isomorphic(og.graph, expected_graph)
    assert og.input_nodes == [0, 3]
    assert og.output_nodes == [100, 200]

    outputs_c = {i for i in og.graph.nodes if i not in og.output_nodes}
    assert og.measurements.keys() == outputs_c
    assert mapping.keys() <= mapping_complete.keys()
    assert set(mapping.values()) <= set(mapping_complete.values())


# Full overlap
def test_compose_3() -> None:
    # Graph 1
    # [0] -- 17 -- (23)
    #        |
    # [3] -- 4  -- (13)
    #
    # Graph 2 = Graph 1
    #
    # Mapping: 0 -> 0, 3 -> 3, 17 -> 17, 4 -> 4, 23 -> 23, 13 -> 13
    #
    # Expected graph = Graph 1

    g: nx.Graph[int]
    g = nx.Graph([(0, 17), (17, 23), (17, 4), (3, 4), (4, 13)])
    inputs = [0, 3]
    outputs = [13, 23]
    meas = {i: Measurement(0, Plane.XY) for i in g.nodes - set(outputs)}
    og_1 = OpenGraph(g, inputs, outputs, meas)

    mapping = {i: i for i in g.nodes}

    og, mapping_complete = og_1.compose(og_1, mapping)

    assert og.isclose(og_1)
    assert mapping.keys() <= mapping_complete.keys()
    assert set(mapping.values()) <= set(mapping_complete.values())


# Overlap inputs/outputs
def test_compose_4() -> None:
    # Graph 1
    # ([17]) -- (3)
    #   |
    #  [18]
    #
    # Graph 2
    # [1] -- 2 -- (3)
    #
    # Mapping: 1 -> 17, 3 -> 300
    #
    # Expected graph
    # (300) -- 2 -- [17] -- (3)
    #                |
    #               [18]

    g: nx.Graph[int]
    g = nx.Graph([(18, 17), (17, 3)])
    inputs = [17, 18]
    outputs = [3, 17]
    meas = {i: Measurement(0, Plane.XY) for i in g.nodes - set(outputs)}
    og_1 = OpenGraph(g, inputs, outputs, meas)

    g = nx.Graph([(1, 2), (2, 3)])
    inputs = [1]
    outputs = [3]
    meas = {i: Measurement(0, Plane.XY) for i in g.nodes - set(outputs)}
    og_2 = OpenGraph(g, inputs, outputs, meas)

    mapping = {1: 17, 3: 300}

    og, mapping_complete = og_1.compose(og_2, mapping)

    expected_graph: nx.Graph[int]
    expected_graph = nx.Graph([(18, 17), (17, 3), (17, 2), (2, 300)])
    assert nx.is_isomorphic(og.graph, expected_graph)
    assert og.input_nodes == [17, 18]  # the input character of node 17 is kept because node 1 (in G2) is an input
    assert og.output_nodes == [
        3,
        300,
    ]  # the output character of node 17 is lost because node 1 (in G2) is not an output

    outputs_c = {i for i in og.graph.nodes if i not in og.output_nodes}
    assert og.measurements.keys() == outputs_c
    assert mapping.keys() <= mapping_complete.keys()
    assert set(mapping.values()) <= set(mapping_complete.values())


# Inverse series composition
def test_compose_5() -> None:
    # Graph 1
    # [1] -- (2)
    #  |
    # [3]
    #
    # Graph 2
    # [3] -- (4)
    #
    # Mapping: 4 -> 1, 3 -> 300
    #
    # Expected graph
    # [300] -- 1 -- (2)
    #          |
    #         [3]

    g: nx.Graph[int]
    g = nx.Graph([(1, 2), (1, 3)])
    inputs = [1, 3]
    outputs = [2]
    meas = {i: Measurement(0, Plane.XY) for i in g.nodes - set(outputs)}
    og_1 = OpenGraph(g, inputs, outputs, meas)

    g = nx.Graph([(3, 4)])
    inputs = [3]
    outputs = [4]
    meas = {i: Measurement(0, Plane.XY) for i in g.nodes - set(outputs)}
    og_2 = OpenGraph(g, inputs, outputs, meas)

    mapping = {4: 1, 3: 300}

    og, mapping_complete = og_1.compose(og_2, mapping)

    expected_graph: nx.Graph[int]
    expected_graph = nx.Graph([(1, 2), (1, 3), (1, 300)])
    assert nx.is_isomorphic(og.graph, expected_graph)
    assert og.input_nodes == [3, 300]
    assert og.output_nodes == [2]

    outputs_c = {i for i in og.graph.nodes if i not in og.output_nodes}
    assert og.measurements.keys() == outputs_c
    assert mapping.keys() <= mapping_complete.keys()
    assert set(mapping.values()) <= set(mapping_complete.values())
