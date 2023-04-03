import unittest
import numpy as np
import networkx as nx
import tests.random_circuit as rc
from graphix.generator import generate_from_graph


class TestGenerator(unittest.TestCase):
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
