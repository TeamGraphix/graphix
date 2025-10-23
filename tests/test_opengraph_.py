from __future__ import annotations

import networkx as nx

from graphix.fundamentals import Plane
from graphix.opengraph_ import OpenGraph


class TestOpenGraph:
    def test_odd_neighbors(self) -> None:
        graph: nx.Graph[int] = nx.Graph([(0, 1), (0, 2), (1, 3), (1, 2), (2, 3), (1, 4)])
        og = OpenGraph(graph=graph, input_nodes=[0], output_nodes=[1, 2, 3, 4], measurements={0: Plane.XY})

        assert og.odd_neighbors([0]) == {1, 2}
        assert og.odd_neighbors([0, 1]) == {0, 1, 3, 4}
        assert og.odd_neighbors([1, 2, 3]) == {4}
        assert og.odd_neighbors([]) == set()

    def test_neighbors(self) -> None:
        graph: nx.Graph[int] = nx.Graph([(0, 1), (0, 2), (1, 3), (1, 2), (2, 3), (1, 4)])
        og = OpenGraph(graph=graph, input_nodes=[0], output_nodes=[1, 2, 3, 4], measurements={0: Plane.XY})

        assert og.neighbors([4]) == {1}
        assert og.neighbors([0, 1]) == {0, 1, 2, 3, 4}
        assert og.neighbors([1, 2, 3]) == {0, 1, 2, 3, 4}
        assert og.neighbors([]) == set()
