import sys
import unittest

from parameterized import parameterized_class

import graphix
from graphix import extraction


@parameterized_class([{"use_rustworkx": False}, {"use_rustworkx": True}])
class TestExtraction(unittest.TestCase):
    def setUp(self):
        if sys.modules.get("rustworkx") is None and self.use_rustworkx is True:
            self.skipTest("rustworkx not installed")

    def test_cluster_extraction_one_ghz_cluster(self):
        gs = graphix.GraphState(use_rustworkx=self.use_rustworkx)
        nodes = [0, 1, 2, 3, 4]
        edges = [(0, 1), (0, 2), (0, 3), (0, 4)]
        gs.add_nodes_from(nodes)
        gs.add_edges_from(edges)
        clusters = extraction.get_fusion_network_from_graph(gs)

        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0] == extraction.ResourceGraph(type=extraction.ResourceType.GHZ, graph=gs), True)

    # we consider everything smaller than 4, a GHZ
    def test_cluster_extraction_small_ghz_cluster_1(self):
        gs = graphix.GraphState(use_rustworkx=self.use_rustworkx)
        nodes = [0, 1, 2]
        edges = [(0, 1), (1, 2)]
        gs.add_nodes_from(nodes)
        gs.add_edges_from(edges)
        clusters = extraction.get_fusion_network_from_graph(gs)

        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0] == extraction.ResourceGraph(type=extraction.ResourceType.GHZ, graph=gs), True)

    # we consider everything smaller than 4, a GHZ
    def test_cluster_extraction_small_ghz_cluster_2(self):
        gs = graphix.GraphState(use_rustworkx=self.use_rustworkx)
        nodes = [0, 1]
        edges = [(0, 1)]
        gs.add_nodes_from(nodes)
        gs.add_edges_from(edges)
        clusters = extraction.get_fusion_network_from_graph(gs)

        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0] == extraction.ResourceGraph(type=extraction.ResourceType.GHZ, graph=gs), True)

    def test_cluster_extraction_one_linear_cluster(self):
        gs = graphix.GraphState(use_rustworkx=self.use_rustworkx)
        nodes = [0, 1, 2, 3, 4, 5, 6]
        edges = [(0, 1), (1, 2), (2, 3), (5, 4), (4, 6), (6, 0)]
        gs.add_nodes_from(nodes)
        gs.add_edges_from(edges)
        clusters = extraction.get_fusion_network_from_graph(gs)

        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0] == extraction.ResourceGraph(type=extraction.ResourceType.LINEAR, graph=gs), True)

    def test_cluster_extraction_one_ghz_one_linear(self):
        gs = graphix.GraphState(use_rustworkx=self.use_rustworkx)
        nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        edges = [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)]
        gs.add_nodes_from(nodes)
        gs.add_edges_from(edges)
        clusters = extraction.get_fusion_network_from_graph(gs)
        self.assertEqual(len(clusters), 2)

        clusters_expected = []
        lin_cluster = graphix.GraphState(use_rustworkx=self.use_rustworkx)
        lin_cluster.add_nodes_from([4, 5, 6, 7, 8, 9])
        lin_cluster.add_edges_from([(4, 5), (5, 6), (6, 7), (7, 8), (8, 9)])
        clusters_expected.append(extraction.ResourceGraph(extraction.ResourceType.LINEAR, lin_cluster))
        ghz_cluster = graphix.GraphState(use_rustworkx=self.use_rustworkx)
        ghz_cluster.add_nodes_from([0, 1, 2, 3, 4])
        ghz_cluster.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4)])
        clusters_expected.append(extraction.ResourceGraph(extraction.ResourceType.GHZ, ghz_cluster))

        self.assertEqual(
            (clusters[0] == clusters_expected[0] and clusters[1] == clusters_expected[1])
            or (clusters[0] == clusters_expected[1] and clusters[1] == clusters_expected[0]),
            True,
        )

    def test_cluster_extraction_pentagonal_cluster(self):
        gs = graphix.GraphState(use_rustworkx=self.use_rustworkx)
        nodes = [0, 1, 2, 3, 4]
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        gs.add_nodes_from(nodes)
        gs.add_edges_from(edges)
        clusters = extraction.get_fusion_network_from_graph(gs)
        self.assertEqual(len(clusters), 2)
        self.assertEqual(
            (clusters[0].type == extraction.ResourceType.GHZ and clusters[1].type == extraction.ResourceType.LINEAR)
            or (clusters[0].type == extraction.ResourceType.LINEAR and clusters[1].type == extraction.ResourceType.GHZ),
            True,
        )
        self.assertEqual(
            (len(clusters[0].graph.nodes) == 3 and len(clusters[1].graph.nodes) == 4)
            or (len(clusters[0].graph.nodes) == 4 and len(clusters[1].graph.nodes) == 3),
            True,
        )

    def test_cluster_extraction_one_plus_two(self):
        gs = graphix.GraphState(use_rustworkx=self.use_rustworkx)
        nodes = [0, 1, 2]
        edges = [(0, 1)]
        gs.add_nodes_from(nodes)
        gs.add_edges_from(edges)
        clusters = extraction.get_fusion_network_from_graph(gs)
        self.assertEqual(len(clusters), 2)
        self.assertEqual(
            (clusters[0].type == extraction.ResourceType.GHZ and clusters[1].type == extraction.ResourceType.GHZ),
            True,
        )
        self.assertEqual(
            (len(clusters[0].graph.nodes) == 2 and len(clusters[1].graph.nodes) == 1)
            or (len(clusters[0].graph.nodes) == 1 and len(clusters[1].graph.nodes) == 2),
            True,
        )
