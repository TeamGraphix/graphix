"""Graph State Resource Extraction
=================================

This module demonstrates how to extract and analyze a resource graph from a
2D cluster state using the GraphState class from the `graphix` library.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import networkx as nx
from graphix import GraphState

if TYPE_CHECKING:
    from collections.abc import Sequence


class GraphStateExtractor:
    """
    Utility class for creating and analyzing graph states.
    """

    extraction_times: list[float]
    equivalence_times: list[float]

    def __init__(self) -> None:
        """Initialize timing logs."""
        self.extraction_times = []
        self.equivalence_times = []

    @staticmethod
    def create_2d_cluster_state(rows: int, cols: int) -> GraphState:
        """
        Create a 2D cluster state as a rectangular lattice.
        """
        gs = GraphState()
        nodes = [i * cols + j for i in range(rows) for j in range(cols)]
        gs.add_nodes_from(nodes)

        edges = []
        for i in range(rows):
            for j in range(cols):
                current = i * cols + j
                if j < cols - 1:
                    edges.append((current, current + 1))
                if i < rows - 1:
                    edges.append((current, current + cols))

        gs.add_edges_from(edges)
        return gs

    def extract_target_graph_state(
        self,
        cluster_state: GraphState,
        target_edges: Sequence[tuple[int, int]],
        target_nodes: Sequence[int] | None = None,
    ) -> tuple[GraphState, list[int]]:
        """
        Extract a graph state using local measurements.
        """
        start = time.perf_counter()

        if target_nodes is None:
            # Use a set to eliminate duplicates; conversion to list handled once
            target_nodes = list({n for edge in target_edges for n in edge})

        target_gs = GraphState()
        target_gs.add_nodes_from(target_nodes)
        target_gs.add_edges_from(target_edges)

        nodes_to_measure = list(set(cluster_state.nodes) - set(target_nodes))

        self.extraction_times.append(time.perf_counter() - start)
        return target_gs, nodes_to_measure

    @staticmethod
    def compute_local_equivalence_metrics(gs: GraphState) -> dict[str, Any]:
        """
        Compute local graph-theoretic properties for analysis.
        """
        graph = nx.Graph(gs.edges)
        degrees = dict(graph.degree)
        degree_sequence = sorted(degrees.values())

        return {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "degree_sequence": degree_sequence,
            "spectrum": sorted(nx.adjacency_spectrum(graph).real.tolist()),
            "triangles": sum(nx.triangles(graph).values()) // 3,
            "is_connected": nx.is_connected(graph),
            "components": nx.number_connected_components(graph),
            "max_degree": max(degree_sequence, default=None),
            "min_degree": min(degree_sequence, default=None),
        }


if __name__ == "__main__":
    extractor = GraphStateExtractor()
    cluster = extractor.create_2d_cluster_state(rows=3, cols=3)

    edges = [(0, 1), (1, 2)]
    target_graph, measured = extractor.extract_target_graph_state(cluster, edges)

    info = extractor.compute_local_equivalence_metrics(target_graph)

    print("\nGraph Analysis:")
    for key, value in info.items():
        print(f"{key.replace('_', ' ').capitalize()}: {value}")
