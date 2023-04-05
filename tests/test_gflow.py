import unittest
import networkx as nx
from graphix.gflow import gflow, flow
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
        input = {1, 2}
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


if __name__ == "__main__":
    unittest.main()
