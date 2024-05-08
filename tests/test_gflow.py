from __future__ import annotations

import itertools
import sys
from typing import TYPE_CHECKING, Dict, NamedTuple, Set, Tuple

import networkx as nx
import pytest
from numpy.random import Generator

from graphix.gflow import find_flow, find_gflow, find_pauliflow, verify_flow, verify_gflow, verify_pauliflow
from tests.random_circuit import get_rand_circuit

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from numpy.random import PCG64


seed = 30


class GraphForTest(NamedTuple):
    graph: nx.Graph
    inputs: set[int]
    outputs: set[int]
    meas_planes: dict[int, str]
    meas_angles: dict[int, float] | None
    label: str
    flow_exist: bool
    gflow_exist: bool
    pauliflow_exist: bool | None


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
        None,
        "no measurement",
        flow_exist=True,
        gflow_exist=True,
        pauliflow_exist=None,
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
        None,
        "line graph with flow and gflow",
        flow_exist=True,
        gflow_exist=True,
        pauliflow_exist=None,
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
        None,
        "graph with flow and gflow",
        flow_exist=True,
        gflow_exist=True,
        pauliflow_exist=None,
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
        None,
        "graph with gflow but no flow",
        flow_exist=False,
        gflow_exist=True,
        pauliflow_exist=None,
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
        None,
        "graph with extended gflow but no flow",
        flow_exist=False,
        gflow_exist=True,
        pauliflow_exist=None,
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
        None,
        "graph with no flow and no gflow",
        flow_exist=False,
        gflow_exist=False,
        pauliflow_exist=None,
    )


def _graph7() -> GraphForTest:
    # graph with no flow or gflow but pauliflow, No.1
    #     3
    #     |
    #     2
    #     |
    # 0 - 1 - 4
    nodes = [0, 1, 2, 3, 4]
    edges = [(0, 1), (1, 2), (1, 4), (2, 3)]
    inputs = {0}
    outputs = {4}
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    meas_planes = {0: "XY", 1: "XY", 2: "XY", 3: "XY"}
    meas_angles = {0: 0.1, 1: 0, 2: 0.1, 3: 0}
    return GraphForTest(
        graph,
        inputs,
        outputs,
        meas_planes,
        meas_angles,
        "graph with no flow and no gflow but pauliflow, No.1",
        flow_exist=False,
        gflow_exist=False,
        pauliflow_exist=True,
    )


def _graph8() -> GraphForTest:
    # graph with no flow or gflow but pauliflow, No.2
    # 1   2   3
    # | /     |
    # 0 - - - 4
    nodes = [0, 1, 2, 3, 4]
    edges = [(0, 1), (0, 2), (0, 4), (3, 4)]
    inputs = {0}
    outputs = {4}
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    meas_planes = {0: "YZ", 1: "XZ", 2: "XY", 3: "YZ"}
    meas_angles = {0: 0.5, 1: 0, 2: 0.5, 3: 0}
    return GraphForTest(
        graph,
        inputs,
        outputs,
        meas_planes,
        meas_angles,
        "graph with no flow and no gflow but pauliflow, No.2",
        flow_exist=False,
        gflow_exist=False,
        pauliflow_exist=True,
    )


def _graph9() -> GraphForTest:
    # graph with no flow or gflow but pauliflow, No.3
    # 0 - 1 -- 3
    #    \|   /|
    #     |\ / |
    #     | /\ |
    #     2 -- 4
    nodes = [0, 1, 2, 3, 4]
    edges = [(0, 1), (0, 4), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    inputs = {0}
    outputs = {3, 4}
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    meas_planes = {0: "YZ", 1: "XZ", 2: "XY"}
    meas_angles = {0: 0.5, 1: 0.1, 2: 0.5}
    return GraphForTest(
        graph,
        inputs,
        outputs,
        meas_planes,
        meas_angles,
        "graph with no flow and no gflow but pauliflow, No.3",
        flow_exist=False,
        gflow_exist=False,
        pauliflow_exist=True,
    )


def generate_test_graphs() -> list[GraphForTest]:
    return [
        _graph1(),
        _graph2(),
        _graph3(),
        _graph4(),
        _graph5(),
        _graph6(),
        _graph7(),
        _graph8(),
        _graph9(),
    ]


if sys.version_info >= (3, 9):
    FlowTestCaseType = dict[str, dict[str, tuple[bool, dict[int, set[int]]]]]
else:
    FlowTestCaseType = Dict[str, Dict[str, Tuple[bool, Dict[int, Set[int]]]]]

FLOW_TEST_CASES: FlowTestCaseType = {
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


GFLOW_TEST_CASES: FlowTestCaseType = {
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

PAULIFLOW_TEST_CASES: FlowTestCaseType = {
    "graph with no flow and no gflow but pauliflow, No.1": {
        "correct pauliflow": (True, {0: {1}, 1: {4}, 2: {3}, 3: {2, 4}}),
        "correct pauliflow 2": (True, {0: {1, 3}, 1: {3, 4}, 2: {3}, 3: {2, 3, 4}}),
        "incorrect pauliflow": (False, {0: {1}, 1: {2}, 2: {3}, 3: {4}}),
        "incorrect pauliflow 2": (False, {0: {1, 3}, 1: {3, 4}, 2: {3, 4}, 3: {2, 3, 4}}),
    },
    "graph with no flow and no gflow but pauliflow, No.2": {
        "correct pauliflow": (True, {0: {0, 1}, 1: {1}, 2: {2}, 3: {4}}),
        "correct pauliflow 2": (True, {0: {0, 1, 2}, 1: {1}, 2: {2}, 3: {1, 2, 4}}),
        "incorrect pauliflow": (False, {0: {1}, 1: {1, 2}, 2: {2, 3}, 3: {4}}),
        "incorrect pauliflow 2": (False, {0: {0}, 1: {1}, 2: {3}, 3: {3}}),
    },
    "graph with no flow and no gflow but pauliflow, No.3": {
        "correct pauliflow": (True, {0: {0, 3, 4}, 1: {1, 2}, 2: {4}}),
        "correct pauliflow 2": (True, {0: {0, 2, 4}, 1: {1, 3}, 2: {2, 3, 4}}),
        "incorrect pauliflow": (False, {0: {0, 3, 4}, 1: {1}, 2: {3, 4}}),
        "incorrect pauliflow 2": (False, {0: {0, 3}, 1: {1, 2, 3}, 2: {2, 3, 4}}),
    },
}

if sys.version_info >= (3, 9):
    FlowTestDataType = tuple[GraphForTest, tuple[bool, dict[int, set[int]]]]
else:
    FlowTestDataType = Tuple[GraphForTest, Tuple[bool, Dict[int, Set[int]]]]


def iterate_compatible(
    graphs: Iterable[GraphForTest],
    cases: FlowTestCaseType,
) -> Iterator[FlowTestDataType]:
    for g in graphs:
        for k, v in cases.items():
            if g.label != k:
                continue
            for vv in v.values():
                yield (g, vv)


class RandomMeasGraph(NamedTuple):
    graph: nx.Graph
    vin: set[int]
    vout: set[int]
    meas_planes: dict[int, str]
    meas_angles: dict[int, float]


def get_rand_graph(rng: Generator, n_nodes: int, edge_prob: float = 0.3) -> RandomMeasGraph:
    graph = nx.Graph()
    nodes = range(n_nodes)
    graph.add_nodes_from(nodes)
    edge_candidates = set(itertools.product(nodes, nodes)) - {(i, i) for i in nodes}
    for edge in edge_candidates:
        if rng.uniform() < edge_prob:
            graph.add_edge(*edge)

    input_nodes_number = rng.integers(1, len(nodes) - 1)
    vin = set(rng.choice(nodes, input_nodes_number, replace=False))
    output_nodes_number = rng.integers(1, len(nodes) - input_nodes_number)
    vout = set(rng.choice(list(set(nodes) - vin), output_nodes_number, replace=False))

    meas_planes = {}
    meas_plane_candidates = ["XY", "XZ", "YZ"]
    meas_angles = {}
    meas_angle_candidates = [0, 0.25, 0.5, 0.75]

    for node in set(graph.nodes()) - vout:
        meas_planes[node] = rng.choice(meas_plane_candidates)
        meas_angles[node] = rng.choice(meas_angle_candidates)

    return RandomMeasGraph(graph, vin, vout, meas_planes, meas_angles)


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
    def test_verify_flow(self, data: FlowTestDataType) -> None:
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
    def test_verify_gflow(self, data: FlowTestDataType) -> None:
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

    @pytest.mark.parametrize("data", iterate_compatible(generate_test_graphs(), PAULIFLOW_TEST_CASES))
    def test_verify_pauliflow(self, data: FlowTestDataType) -> None:
        test_graph, test_case = data
        expected, pauliflow = test_case
        angles = test_graph.meas_angles
        assert angles is not None

        valid = verify_pauliflow(
            test_graph.graph,
            test_graph.inputs,
            test_graph.outputs,
            pauliflow,
            test_graph.meas_planes,
            angles,
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

    @pytest.mark.parametrize("jumps", range(1, 51))
    def test_rand_graph_flow(self, fx_bg: PCG64, jumps: int) -> None:
        # test finding algorithm and verification for random graphs
        rng = Generator(fx_bg.jumped(jumps))
        n_nodes = 5
        graph, vin, vout, meas_planes, meas_angles = get_rand_graph(rng, n_nodes)
        f, l_k = find_flow(graph, vin, vout, meas_planes)
        if f:
            valid = verify_flow(graph, vin, vout, f, meas_planes)
            assert valid

    @pytest.mark.parametrize("jumps", range(1, 51))
    def test_rand_graph_gflow(self, fx_bg: PCG64, jumps: int) -> None:
        # test finding algorithm and verification for random graphs
        rng = Generator(fx_bg.jumped(jumps))
        n_nodes = 5
        graph, vin, vout, meas_planes, meas_angles = get_rand_graph(rng, n_nodes)

        g, l_k = find_gflow(graph, vin, vout, meas_planes)
        if g:
            valid = verify_gflow(graph, vin, vout, g, meas_planes)
            assert valid

    @pytest.mark.parametrize("jumps", range(1, 51))
    def test_rand_graph_pauliflow(self, fx_bg: PCG64, jumps: int) -> None:
        # test finding algorithm and verification for random graphs
        rng = Generator(fx_bg.jumped(jumps))
        n_nodes = 5
        graph, vin, vout, meas_planes, meas_angles = get_rand_graph(rng, n_nodes)

        p, l_k = find_pauliflow(graph, vin, vout, meas_planes, meas_angles)
        if p:
            valid = verify_pauliflow(graph, vin, vout, p, meas_planes, meas_angles)
            assert valid
