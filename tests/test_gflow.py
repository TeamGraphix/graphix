import unittest
import networkx as nx
from graphix.gflow import gflow, flow, generate_from_graph
import tests.random_circuit as rc
import numpy as np


class TestGflow(unittest.TestCase):
    def test_flow(self):
        nodes = [i for i in range(9)]
        edges = [(0, 3), (1, 4), (2, 5), (1, 3), (2, 4), (3, 6), (4, 7), (5, 8)]
        input = set()
        output = {6, 7, 8}
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        f, l_k = flow(G, input, output)
        expected_f = {0: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8}
        expected_lk = {0: 4, 1: 3, 2: 2, 3: 1, 4: 1, 5: 1, 6: 0, 7: 0, 8: 0}
        self.assertEqual(f, expected_f)
        self.assertEqual(l_k, expected_lk)

    def test_gflow(self):
        nodes = [i for i in range(9)]
        edges = [(0, 3), (1, 4), (2, 5), (1, 3), (2, 4), (3, 6), (4, 7), (5, 8)]
        input = set()
        output = {6, 7, 8}
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        g, l_k = gflow(G, input, output)
        expected_g = {0: {3, 4, 5}, 1: {4, 5}, 2: {5}, 3: {6}, 4: {7}, 5: {8}}
        expected_lk = {0: 2, 1: 2, 2: 2, 3: 1, 4: 1, 5: 1, 6: 0, 7: 0, 8: 0}
        self.assertEqual(g, expected_g)
        self.assertEqual(l_k, expected_lk)

    def test_noflow(self):
        nodes = [i for i in range(1, 13)]
        edges = [
            (1, 3),
            (3, 6),
            (6, 9),
            (9, 11),
            (3, 4),
            (6, 7),
            (4, 7),
            (4, 5),
            (7, 8),
            (2, 5),
            (5, 8),
            (8, 10),
            (10, 12),
        ]
        input = set()
        output = {11, 12}
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        f, l_k = flow(G, input, output)
        self.assertIsNone(f)
        self.assertIsNone(l_k)

    def test_nogflow(self):
        nodes = [i for i in range(1, 13)]
        edges = [
            (1, 3),
            (3, 6),
            (6, 9),
            (9, 11),
            (3, 4),
            (6, 7),
            (4, 7),
            (4, 5),
            (7, 8),
            (2, 5),
            (5, 8),
            (8, 10),
            (10, 12),
        ]
        input = set()
        output = {11, 12}
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        g, l_k = gflow(G, input, output)
        self.assertIsNone(g)
        self.assertIsNone(l_k)

    def test_pattern_generation_flow(self):
        nqubits = 3
        depth = 2
        pairs = [(0, 1), (1, 2)]
        circuit = rc.generate_gate(nqubits, depth, pairs)
        # transpile into graph
        pattern = circuit.transpile()
        pattern.standardize()
        pattern.shift_signals()
        # get the graph and generate pattern again with flow algorithm
        nodes, edges = pattern.get_graph()
        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        input = [0, 1, 2]
        angles = dict()
        for cmd in pattern.get_measurement_order():
            angles[cmd[1]] = cmd[3]
        pattern2 = generate_from_graph(g, angles, input, pattern.output_nodes)
        # check that the new one runs and returns correct result
        pattern2.standardize()
        pattern2.shift_signals()
        pattern2.minimize_space()
        state = circuit.simulate_statevector()
        state_mbqc = pattern2.simulate_pattern()
        np.testing.assert_almost_equal(np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())), 1)


if __name__ == "__main__":
    unittest.main()
