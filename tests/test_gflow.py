# %%
from __future__ import annotations

import unittest

import networkx as nx
import numpy as np
from itertools import product
from graphix.gflow import (
    find_flow,
    find_gflow,
    find_pauliflow,
    get_input_from_flow,
    verify_flow,
    verify_gflow,
    verify_pauliflow,
)

from tests.random_circuit import get_rand_circuit

seed = 30


class GraphForTest:
    def __init__(
        self,
        graph: nx.Graph,
        inputs: set,
        outputs: set,
        meas_planes: dict[int, set],
        meas_angles: dict[int, float],
        flow_exist: bool,
        gflow_exist: bool,
        pauliflow_exist: bool,
        label: str,
    ):
        self.graph = graph
        self.inputs = inputs
        self.outputs = outputs
        self.meas_planes = meas_planes
        self.meas_angles = meas_angles
        self.flow_exist = flow_exist
        self.gflow_exist = gflow_exist
        self.pauliflow_exist = pauliflow_exist
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
    test_graph = GraphForTest(graph, inputs, outputs, meas_planes, None, True, True, None, "no measurement")
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
        None,
        True,
        True,
        None,
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
    test_graph = GraphForTest(graph, inputs, outputs, meas_planes, None, True, True, None, "graph with flow and gflow")
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
    test_graph = GraphForTest(
        graph, inputs, outputs, meas_planes, None, False, True, None, "graph with gflow but no flow"
    )
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
    test_graph = GraphForTest(
        graph,
        inputs,
        outputs,
        meas_planes,
        None,
        False,
        True,
        None,
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
        None,
        False,
        False,
        None,
        "graph with no flow and no gflow",
    )
    graphs.append(test_graph)

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
    test_graph = GraphForTest(
        graph,
        inputs,
        outputs,
        meas_planes,
        meas_angles,
        False,
        False,
        True,
        "graph with no flow and no gflow but pauliflow, No.1",
    )
    graphs.append(test_graph)

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
    test_graph = GraphForTest(
        graph,
        inputs,
        outputs,
        meas_planes,
        meas_angles,
        False,
        False,
        True,
        "graph with no flow and no gflow but pauliflow, No.2",
    )
    graphs.append(test_graph)

    graphs.append(test_graph)

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
    test_graph = GraphForTest(
        graph,
        inputs,
        outputs,
        meas_planes,
        meas_angles,
        False,
        False,
        True,
        "graph with no flow and no gflow but pauliflow, No.3",
    )
    graphs.append(test_graph)

    return graphs


def get_rand_graph(n_nodes, edge_prob=0.3, seed=None):
    np.random.seed(seed)
    graph = nx.Graph()
    nodes = list(range(n_nodes))
    graph.add_nodes_from(nodes)
    edge_candidates = set(list(product(nodes, nodes))) - set([(i, i) for i in nodes])
    for edge in edge_candidates:
        if np.random.random() < 0.3:
            graph.add_edge(*edge)

    input_nodes_number = np.random.randint(1, len(nodes) - 1)
    vin = set(np.random.choice(nodes, input_nodes_number, replace=False))
    output_nodes_number = np.random.randint(1, len(nodes) - input_nodes_number)
    vout = set(np.random.choice(list(set(nodes) - vin), output_nodes_number, replace=False))

    meas_planes = dict()
    meas_plane_candidates = ["XY", "XZ", "YZ"]
    meas_angles = dict()
    meas_angle_candidates = [0, 0.25, 0.5, 0.75]
    for node in set(graph.nodes()) - vout:
        meas_planes[node] = np.random.choice(meas_plane_candidates)
        meas_angles[node] = np.random.choice(meas_angle_candidates)

    return graph, vin, vout, meas_planes, meas_angles


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

    def test_pauliflow(self):
        test_graphs = generate_test_graphs()
        for test_graph in test_graphs:
            if test_graph.meas_angles is None:
                continue
            with self.subTest(test_graph.label):
                p, l_k = find_pauliflow(
                    test_graph.graph,
                    test_graph.inputs,
                    test_graph.outputs,
                    test_graph.meas_planes,
                    test_graph.meas_angles,
                )
                self.assertEqual(test_graph.pauliflow_exist, p is not None)

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

    def test_verify_pauliflow(self):
        pauliflow_test_cases = dict()
        pauliflow_test_cases["graph with no flow and no gflow but pauliflow, No.1"] = {
            "correct pauliflow": (True, {0: {1}, 1: {4}, 2: {3}, 3: {2, 4}}),
            "correct pauliflow 2": (True, {0: {1, 3}, 1: {3, 4}, 2: {3}, 3: {2, 3, 4}}),
            "incorrect pauliflow": (False, {0: {1}, 1: {2}, 2: {3}, 3: {4}}),
            "incorrect pauliflow 2": (False, {0: {1, 3}, 1: {3, 4}, 2: {3, 4}, 3: {2, 3, 4}}),
        }
        pauliflow_test_cases["graph with no flow and no gflow but pauliflow, No.2"] = {
            "correct pauliflow": (True, {0: {0, 1}, 1: {1}, 2: {2}, 3: {4}}),
            "correct pauliflow 2": (True, {0: {0, 1, 2}, 1: {1}, 2: {2}, 3: {1, 2, 4}}),
            "incorrect pauliflow": (False, {0: {1}, 1: {1, 2}, 2: {2, 3}, 3: {4}}),
            "incorrect pauliflow 2": (False, {0: {0}, 1: {1}, 2: {3}, 3: {3}}),
        }
        pauliflow_test_cases["graph with no flow and no gflow but pauliflow, No.3"] = {
            "correct pauliflow": (True, {0: {0, 3, 4}, 1: {1, 2}, 2: {4}}),
            "correct pauliflow 2": (True, {0: {0, 2, 4}, 1: {1, 3}, 2: {2, 3, 4}}),
            "incorrect pauliflow": (False, {0: {0, 3, 4}, 1: {1}, 2: {3, 4}}),
            "incorrect pauliflow 2": (False, {0: {0, 3}, 1: {1, 2, 3}, 2: {2, 3, 4}}),
        }

        test_graphs = generate_test_graphs()
        for test_graph in test_graphs:
            if test_graph.label not in pauliflow_test_cases:
                continue
            with self.subTest(test_graph.label):
                for test_case, (expected, pauliflow) in pauliflow_test_cases[test_graph.label].items():
                    with self.subTest([test_graph.label, test_case]):
                        valid = verify_pauliflow(
                            test_graph.graph,
                            test_graph.inputs,
                            test_graph.outputs,
                            pauliflow,
                            test_graph.meas_planes,
                            test_graph.meas_angles,
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

    def test_rand_graph(self):
        # test finding algorithm and verification for random graphs
        n_nodes = 5
        for i in range(50):
            graph, vin, vout, meas_planes, meas_angles = get_rand_graph(n_nodes, seed=seed)
            f, l_k = find_flow(graph, vin, vout, meas_planes)
            if f:
                valid = verify_flow(graph, vin, vout, f, meas_planes)
                self.assertEqual(True, valid)

            g, l_k = find_gflow(graph, vin, vout, meas_planes)
            if g:
                valid = verify_gflow(graph, vin, vout, g, meas_planes)
                self.assertEqual(True, valid)

            p, l_k = find_pauliflow(graph, vin, vout, meas_planes, meas_angles)
            if p:
                valid = verify_pauliflow(graph, vin, vout, p, meas_planes, meas_angles)
                self.assertEqual(True, valid)


if __name__ == "__main__":
    unittest.main()
