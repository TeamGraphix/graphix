import unittest
import numpy as np
import networkx as nx
import tests.random_circuit as rc
from graphix.generator import generate_from_graph


class TestGenerator(unittest.TestCase):
    def test_pattern_generation_determinism_flow(self):
        graph = nx.Graph([(0, 3), (1, 4), (2, 5), (1, 3), (2, 4), (3, 6), (4, 7), (5, 8)])
        inputs = {0, 1, 2}
        outputs = {6, 7, 8}
        angles = np.random.randn(6)
        results = []
        repeats = 3  # for testing the determinism of a pattern
        for _ in range(repeats):
            pattern = generate_from_graph(graph, angles, inputs, outputs)
            pattern.standardize()
            pattern.minimize_space()
            state = pattern.simulate_pattern()
            results.append(state)
        combinations = [(0, 1), (0, 2), (1, 2)]
        for i, j in combinations:
            inner_product = np.dot(results[i].flatten(), results[j].flatten().conjugate())
            np.testing.assert_almost_equal(abs(inner_product), 1)

    def test_pattern_generation_determinism_gflow(self):
        graph = nx.Graph([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (3, 6), (1, 6)])
        inputs = {1, 3, 5}
        outputs = {2, 4, 6}
        angles = np.random.randn(11)
        results = []
        repeats = 3  # for testing the determinism of a pattern
        for _ in range(repeats):
            pattern = generate_from_graph(graph, angles, inputs, outputs)
            pattern.standardize()
            pattern.minimize_space()
            state = pattern.simulate_pattern()
            results.append(state)
        combinations = [(0, 1), (0, 2), (1, 2)]
        for i, j in combinations:
            inner_product = np.dot(results[i].flatten(), results[j].flatten().conjugate())
            np.testing.assert_almost_equal(abs(inner_product), 1)

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
        for cmd in pattern.get_measurement_commands():
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
