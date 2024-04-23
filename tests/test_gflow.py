from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import networkx as nx
import pytest

from graphix.gflow import find_flow, find_gflow, verify_flow, verify_gflow
from tests.random_circuit import get_rand_circuit

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from numpy.random import Generator

seed = 30


class GraphForTest(NamedTuple):
    graph: nx.Graph
    inputs: set[int]
    outputs: set[int]
    meas_planes: dict[int, str]
    label: str
    flow_exist: bool
    gflow_exist: bool


def _graph1() -> GraphForTest:
    # no measurement
    # 1
    # |
    # 2
    nodes = [1, 2]
    edges = [(1, 2)]
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    inputs = {1, 2}
    outputs = {1, 2}
    meas_planes = {}
    return GraphForTest(
        graph,
        inputs,
        outputs,
        meas_planes,
        "no measurement",
        flow_exist=True,
        gflow_exist=True,
    )


def _graph2() -> GraphForTest:
    # line graph with flow and gflow
    # 1 - 2 - 3 - 4 - 5
    nodes = [1, 2, 3, 4, 5]
    edges = [(1, 2), (2, 3), (3, 4), (4, 5)]
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    inputs = {1}
    outputs = {5}
    meas_planes = {1: "XY", 2: "XY", 3: "XY", 4: "XY"}
    return GraphForTest(
        graph,
        inputs,
        outputs,
        meas_planes,
        "line graph with flow and gflow",
        flow_exist=True,
        gflow_exist=True,
    )


def _graph3() -> GraphForTest:
    # graph with flow and gflow
    # 1 - 3 - 5
    #     |
    # 2 - 4 - 6
    nodes = [1, 2, 3, 4, 5, 6]
    edges = [(1, 3), (2, 4), (3, 4), (3, 5), (4, 6)]
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    inputs = {1, 2}
    outputs = {5, 6}
    meas_planes = {1: "XY", 2: "XY", 3: "XY", 4: "XY"}
    return GraphForTest(
        graph,
        inputs,
        outputs,
        meas_planes,
        "graph with flow and gflow",
        flow_exist=True,
        gflow_exist=True,
    )


def _graph4() -> GraphForTest:
    # graph with gflow but flow
    #   ______
    #  /      |
    # 1 - 4   |
    #    /    |
    #   /     |
    #  /      |
    # 2 - 5   |
    #  \ /    |
    #   X    /
    #  / \  /
    # 3 - 6
    nodes = [1, 2, 3, 4, 5, 6]
    edges = [(1, 4), (1, 6), (2, 4), (2, 5), (2, 6), (3, 5), (3, 6)]
    inputs = {1, 2, 3}
    outputs = {4, 5, 6}
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    meas_planes = {1: "XY", 2: "XY", 3: "XY"}
    return GraphForTest(
        graph,
        inputs,
        outputs,
        meas_planes,
        "graph with gflow but no flow",
        flow_exist=False,
        gflow_exist=True,
    )


def _graph5() -> GraphForTest:
    # graph with extended gflow but flow
    #   0 - 1
    #  /|   |
    # 4 |   |
    #  \|   |
    #   2 - 5 - 3
    nodes = [0, 1, 2, 3, 4, 5]
    edges = [(0, 1), (0, 2), (0, 4), (1, 5), (2, 4), (2, 5), (3, 5)]
    inputs = {0, 1}
    outputs = {4, 5}
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    meas_planes = {0: "XY", 1: "XY", 2: "ZX", 3: "YZ"}
    return GraphForTest(
        graph,
        inputs,
        outputs,
        meas_planes,
        "graph with extended gflow but no flow",
        flow_exist=False,
        gflow_exist=True,
    )


def _graph6() -> GraphForTest:
    # graph with no flow and no gflow
    # 1 - 3
    #  \ /
    #   X
    #  / \
    # 2 - 4
    nodes = [1, 2, 3, 4]
    edges = [(1, 3), (1, 4), (2, 3), (2, 4)]
    inputs = {1, 2}
    outputs = {3, 4}
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    meas_planes = {1: "XY", 2: "XY"}
    return GraphForTest(
        graph,
        inputs,
        outputs,
        meas_planes,
        "graph with no flow and no gflow",
        flow_exist=False,
        gflow_exist=False,
    )


def generate_test_graphs() -> list[GraphForTest]:
    return [
        _graph1(),
        _graph2(),
        _graph3(),
        _graph4(),
        _graph5(),
        _graph6(),
    ]


TestCaseType = dict[str, dict[str, tuple[bool, dict[int, set[int]]]]]

FLOW_TEST_CASES: TestCaseType = {
    "no measurement": {
        "empty flow": (True, {}),
        "measure output": (False, {1: {2}}),
    },
    "line graph with flow and gflow": {
        "correct flow": (True, {1: {2}, 2: {3}, 3: {4}, 4: {5}}),
        "acausal flow": (False, {1: {3}, 3: {2, 4}, 2: {1}, 4: {5}}),
        "gflow": (False, {1: {2, 5}, 2: {3, 5}, 3: {4, 5}, 4: {5}}),
    },
    "graph with flow and gflow": {
        "correct flow": (True, {1: {3}, 2: {4}, 3: {5}, 4: {6}}),
        "acausal flow": (False, {1: {4}, 2: {3}, 3: {4}, 4: {1}}),
        "gflow": (False, {1: {3, 5}, 2: {4, 5}, 3: {5, 6}, 4: {6}}),
    },
}


GFLOW_TEST_CASES: TestCaseType = {
    "no measurement": {
        "empty flow": (True, {}),
        "measure output": (False, {1: {2}}),
    },
    "line graph with flow and gflow": {
        "correct flow": (True, {1: {2}, 2: {3}, 3: {4}, 4: {5}}),
        "acausal flow": (False, {1: {3}, 3: {2, 4}, 2: {1}, 4: {5}}),
        "gflow": (True, {1: {2, 5}, 2: {3, 5}, 3: {4, 5}, 4: {5}}),
    },
    "graph with flow and gflow": {
        "correct flow": (True, {1: {3}, 2: {4}, 3: {5}, 4: {6}}),
        "acausal flow": (False, {1: {4}, 2: {3}, 3: {4}, 4: {1}}),
        "gflow": (True, {1: {3, 5}, 2: {4, 5}, 3: {5, 6}, 4: {6}}),
    },
    "graph with extended gflow but no flow": {
        "correct gflow": (
            True,
            {0: {1, 2, 3, 4}, 1: {2, 3, 4, 5}, 2: {2, 4}, 3: {3}},
        ),
        "correct gflow 2": (True, {0: {1, 2, 4}, 1: {3, 5}, 2: {2, 4}, 3: {3}}),
        "incorrect gflow": (
            False,
            {0: {1, 2, 3, 4}, 1: {2, 3, 4, 5}, 2: {2, 4}, 3: {3, 4}},
        ),
        "incorrect gflow 2": (
            False,
            {0: {1, 3, 4}, 1: {2, 3, 4, 5}, 2: {2, 4}, 3: {3}},
        ),
    },
}

TestDataType = tuple[GraphForTest, tuple[bool, dict[int, set[int]]]]


def iterate_compatible(
    graphs: Iterable[GraphForTest],
    cases: TestCaseType,
) -> Iterator[TestDataType]:
    for g in graphs:
        for k, v in cases.items():
            if g.label != k:
                continue
            for vv in v.values():
                yield (g, vv)


class TestGflow:
    @pytest.mark.parametrize("test_graph", generate_test_graphs())
    def test_flow(self, test_graph: GraphForTest) -> None:
        f, l_k = find_flow(
            test_graph.graph,
            test_graph.inputs,
            test_graph.outputs,
            test_graph.meas_planes,
        )
        assert test_graph.flow_exist == (f is not None)

    @pytest.mark.parametrize("test_graph", generate_test_graphs())
    def test_gflow(self, test_graph: GraphForTest) -> None:
        g, l_k = find_gflow(
            test_graph.graph,
            test_graph.inputs,
            test_graph.outputs,
            test_graph.meas_planes,
        )
        assert test_graph.gflow_exist == (g is not None)

    @pytest.mark.parametrize("data", iterate_compatible(generate_test_graphs(), FLOW_TEST_CASES))
    def test_verify_flow(self, data: TestDataType) -> None:
        test_graph, test_case = data
        expected, flow = test_case
        valid = verify_flow(
            test_graph.graph,
            test_graph.inputs,
            test_graph.outputs,
            flow,
            test_graph.meas_planes,
        )
        assert expected == valid

    @pytest.mark.parametrize("data", iterate_compatible(generate_test_graphs(), GFLOW_TEST_CASES))
    def test_verify_gflow(self, data: TestDataType) -> None:
        test_graph, test_case = data
        expected, gflow = test_case

        valid = verify_gflow(
            test_graph.graph,
            test_graph.inputs,
            test_graph.outputs,
            gflow,
            test_graph.meas_planes,
        )
        assert expected == valid

    def test_with_rand_circ(self, fx_rng: Generator) -> None:
        # test for large graph
        # graph transpiled from circuit always has a flow
        circ = get_rand_circuit(10, 10, fx_rng)
        pattern = circ.transpile().pattern
        nodes, edges = pattern.get_graph()
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        input_ = set(pattern.input_nodes)
        output = set(pattern.output_nodes)
        meas_planes = pattern.get_meas_plane()
        f, l_k = find_flow(graph, input_, output, meas_planes)
        valid = verify_flow(graph, input_, output, f, meas_planes)

        assert valid

    # TODO: Remove after fixed
    @pytest.mark.skip()
    def test_rand_circ_gflow(self, fx_rng: Generator) -> None:
        # test for large graph
        # pauli-node measured graph always has gflow
        circ = get_rand_circuit(5, 5, fx_rng)
        pattern = circ.transpile().pattern
        pattern.standardize()
        pattern.shift_signals()
        pattern.perform_pauli_measurements()
        nodes, edges = pattern.get_graph()
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        input_ = set()
        output = set(pattern.output_nodes)
        meas_planes = pattern.get_meas_plane()
        g, l_k = find_gflow(graph, input_, output, meas_planes)

        valid = verify_gflow(graph, input_, output, g, meas_planes)

        assert valid
