from __future__ import annotations

import unittest

import networkx as nx
from graphix.gflow import flow, gflow, check_flow, check_gflow


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
    #   X     |
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
    meas_planes = {0: "XY", 1: "XY", 2: "XZ", 3: "YZ"}
    test_graph = GraphForTest(graph, inputs, outputs, meas_planes, False, True, "graph with gflow but no flow")
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
                f, l_k = flow(
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
                g, l_k = gflow(
                    test_graph.graph,
                    test_graph.inputs,
                    test_graph.outputs,
                    test_graph.meas_planes,
                )
                self.assertEqual(test_graph.gflow_exist, g is not None)

    def test_check_flow(self):
        flow_test_cases = dict()
        flow_test_cases["no measurement"] = {
            "empty flow": (True, dict()),
            "measure output": (False, {1: {2}}),
        }
        flow_test_cases["line graph with flow and gflow"] = {
            "correct flow": (True, {1: {2}, 2: {3}, 3: {4}, 4: {5}}),
            "acausal flow": (False, {1: {3}, 3: {2}, 2: {5}, 4: {5}}),
            "gflow": (False, {1: {2, 5}, 2: {3, 5}, 3: {4, 5}, 4: {5}}),
        }
        flow_test_cases["graph with flow and gflow"] = {
            "correct flow": (True, {1: {3}, 2: {4}, 3: {5}, 4: {6}}),
            "acausal flow": (False, {1: {4}, 2: {3}, 3: {4}, 4: {1}}),
            "gflow": (False, {1: {3, 5}, 2: {4, 5}, 3: {4, 6}, 4: {6}}),
        }

        test_graphs = generate_test_graphs()
        for test_graph in test_graphs:
            if test_graph.label not in flow_test_cases:
                continue
            with self.subTest(test_graph.label):
                for test_case, (expected, flow) in flow_test_cases[test_graph.label].items():
                    with self.subTest(test_case):
                        valid = check_flow(test_graph.graph, flow, test_graph.meas_planes)
                        self.assertEqual(expected, valid)

    def test_check_gflow(self):
        pass


if __name__ == "__main__":
    unittest.main()
