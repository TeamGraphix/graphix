from __future__ import annotations

import networkx as nx

from graphix.fundamentals import Plane
from graphix.opengraph_ import OpenGraph


class TestFlow:

    # Simple flow tests
    def test_causal_flow_0(self) -> None:
        og = get_linear_og()
        flow = og.find_causal_flow()

        assert flow is not None
        assert flow.correction_function == {0: {1}, 1: {2}, 2: {3}}
        assert flow.partial_order_layers == [{3}, {2}, {1}, {0}]


def get_linear_og() -> OpenGraph[Plane]:
    """Return linear open graph with causal flow."""
    graph: nx.Graph[int] = nx.Graph([(0, 1), (1, 2), (2, 3)])
    input_nodes = [0]
    output_nodes = [3]
    measurements = dict.fromkeys(range(3), Plane.XY)

    return OpenGraph(graph, measurements, input_nodes, output_nodes)
