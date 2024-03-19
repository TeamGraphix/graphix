# %%
from __future__ import annotations

import unittest

import networkx as nx
import numpy as np
from graphix.gflow import find_flow, find_gflow, get_input_from_flow, verify_flow, verify_gflow

from tests.random_circuit import get_rand_circuit

seed = 30


class GraphForTest:
    def __init__(
        self,
        graph: nx.Graph,
        inputs: set,
        outputs: set,
        meas_planes: dict[int, set],
        flow_exist: bool,
        gflow_exist: bool,
        label: str,
    ):
        self.graph = graph
        self.inputs = inputs
        self.outputs = outputs
        self.meas_planes = meas_planes
        self.flow_exist = flow_exist
        self.gflow_exist = gflow_exist
        self.label = label


def generate_test_graphs() -> list[GraphForTest]:
    graphs = []

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
    test_graph = GraphForTest(graph, inputs, outputs, meas_planes, True, True, "no measurement")
    graphs.append(test_graph)

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
    test_graph = GraphForTest(
        graph,
        inputs,
        outputs,
        meas_planes,
        True,
        True,
        "line graph with flow and gflow",
    )
    graphs.append(test_graph)

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
    test_graph = GraphForTest(graph, inputs, outputs, meas_planes, True, True, "graph with flow and gflow")
    graphs.append(test_graph)

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
    test_graph = GraphForTest(graph, inputs, outputs, meas_planes, False, True, "graph with gflow but no flow")
    graphs.append(test_graph)

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
    test_graph = GraphForTest(
        graph,
        inputs,
        outputs,
        meas_planes,
        False,
        True,
        "graph with extended gflow but no flow",
    )
    graphs.append(test_graph)

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
    test_graph = GraphForTest(
        graph,
        inputs,
        outputs,
        meas_planes,
        False,
        False,
        "graph with no flow and no gflow",
    )
    graphs.append(test_graph)

    return graphs


class TestGflow(unittest.TestCase):
    def test_flow(self):
        test_graphs = generate_test_graphs()
        for test_graph in test_graphs:
            with self.subTest(test_graph.label):
                f, l_k = find_flow(
                    test_graph.graph,
                    test_graph.inputs,
                    test_graph.outputs,
                    test_graph.meas_planes,
                )
                self.assertEqual(test_graph.flow_exist, f is not None)

    def test_gflow(self):
        test_graphs = generate_test_graphs()
        for test_graph in test_graphs:
            with self.subTest(test_graph.label):
                g, l_k = find_gflow(
                    test_graph.graph,
                    test_graph.inputs,
                    test_graph.outputs,
                    test_graph.meas_planes,
                )
                self.assertEqual(test_graph.gflow_exist, g is not None)

    def test_verify_flow(self):
        flow_test_cases = dict()
        flow_test_cases["no measurement"] = {
            "empty flow": (True, dict()),
            "measure output": (False, {1: {2}}),
        }
        flow_test_cases["line graph with flow and gflow"] = {
            "correct flow": (True, {1: {2}, 2: {3}, 3: {4}, 4: {5}}),
            "acausal flow": (False, {1: {3}, 3: {2, 4}, 2: {1}, 4: {5}}),
            "gflow": (False, {1: {2, 5}, 2: {3, 5}, 3: {4, 5}, 4: {5}}),
        }
        flow_test_cases["graph with flow and gflow"] = {
            "correct flow": (True, {1: {3}, 2: {4}, 3: {5}, 4: {6}}),
            "acausal flow": (False, {1: {4}, 2: {3}, 3: {4}, 4: {1}}),
            "gflow": (False, {1: {3, 5}, 2: {4, 5}, 3: {5, 6}, 4: {6}}),
        }

        test_graphs = generate_test_graphs()
        for test_graph in test_graphs:
            if test_graph.label not in flow_test_cases:
                continue
            with self.subTest(test_graph.label):
                for test_case, (expected, flow) in flow_test_cases[test_graph.label].items():
                    with self.subTest([test_graph.label, test_case]):
                        valid = verify_flow(
                            test_graph.graph,
                            test_graph.inputs,
                            test_graph.outputs,
                            flow,
                            test_graph.meas_planes,
                        )
                        self.assertEqual(expected, valid)

    def test_verify_gflow(self):
        gflow_test_cases = dict()
        gflow_test_cases["no measurement"] = {
            "empty flow": (True, dict()),
            "measure output": (False, {1: {2}}),
        }
        gflow_test_cases["line graph with flow and gflow"] = {
            "correct flow": (True, {1: {2}, 2: {3}, 3: {4}, 4: {5}}),
            "acausal flow": (False, {1: {3}, 3: {2, 4}, 2: {1}, 4: {5}}),
            "gflow": (True, {1: {2, 5}, 2: {3, 5}, 3: {4, 5}, 4: {5}}),
        }
        gflow_test_cases["graph with flow and gflow"] = {
            "correct flow": (True, {1: {3}, 2: {4}, 3: {5}, 4: {6}}),
            "acausal flow": (False, {1: {4}, 2: {3}, 3: {4}, 4: {1}}),
            "gflow": (True, {1: {3, 5}, 2: {4, 5}, 3: {5, 6}, 4: {6}}),
        }
        gflow_test_cases["graph with extended gflow but no flow"] = {
            "correct gflow": (
                True,
                {0: {1, 2, 3, 4}, 1: {2, 3, 4, 5}, 2: {2, 4}, 3: {3}},
            ),
            "correct glow 2": (True, {0: {1, 2, 4}, 1: {3, 5}, 2: {2, 4}, 3: {3}}),
            "incorrect gflow": (
                False,
                {0: {1, 2, 3, 4}, 1: {2, 3, 4, 5}, 2: {2, 4}, 3: {3, 4}},
            ),
            "incorrect gflow 2": (
                False,
                {0: {1, 3, 4}, 1: {2, 3, 4, 5}, 2: {2, 4}, 3: {3}},
            ),
        }

        test_graphs = generate_test_graphs()
        for test_graph in test_graphs:
            if test_graph.label not in gflow_test_cases:
                continue
            with self.subTest(test_graph.label):
                for test_case, (expected, gflow) in gflow_test_cases[test_graph.label].items():
                    with self.subTest([test_graph.label, test_case]):
                        valid = verify_gflow(
                            test_graph.graph,
                            test_graph.inputs,
                            test_graph.outputs,
                            gflow,
                            test_graph.meas_planes,
                        )
                        self.assertEqual(expected, valid)

    def test_with_rand_circ(self):
        # test for large graph
        # graph transpiled from circuit always has a flow
        circ = get_rand_circuit(10, 10, seed=seed)
        pattern = circ.transpile()
        nodes, edges = pattern.get_graph()
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        input = set(pattern.input_nodes)
        output = set(pattern.output_nodes)
        meas_planes = pattern.get_meas_plane()
        f, l_k = find_flow(graph, input, output, meas_planes)
        valid = verify_flow(graph, input, output, f, meas_planes)

        self.assertEqual(True, valid)

    def test_rand_circ_gflow(self):
        # test for large graph
        # pauli-node measured graph always has gflow
        circ = get_rand_circuit(5, 5, seed=seed)
        pattern = circ.transpile()
        pattern.standardize()
        pattern.shift_signals()
        pattern.perform_pauli_measurements()
        nodes, edges = pattern.get_graph()
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        input = set()
        output = set(pattern.output_nodes)
        meas_planes = pattern.get_meas_plane()
        g, l_k = find_gflow(graph, input, output, meas_planes)

        valid = verify_gflow(graph, input, output, g, meas_planes)

        self.assertEqual(True, valid)


if __name__ == "__main__":
    unittest.main()
