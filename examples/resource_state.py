"""
Graph State Resource Extraction
===============================

This module demonstrates how to extract and analyze a resource graph from a
2D cluster state using the GraphState class from the `graphix` library.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import networkx as nx
from graphix import GraphState

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class ResourceGraphInfo:
    """
    Container for resource graph metrics.

    Attributes
    ----------
    graph_class : str or None
        Class name of the graph object.
    attributes : list of str
        Extra attributes for descriptive purposes.
    nodes : int or None
        Number of nodes in the graph.
    edges : int or None
        Number of edges in the graph.
    kind : str or None
        Kind of graph or state.
    resource_type : str or None
        Type of resource (renamed from `type` to avoid conflict).
    degree_sequence : list of int
        Sorted sequence of node degrees.
    spectrum : list of float
        Adjacency spectrum of the graph.
    triangles : int or None
        Total number of triangles in the graph.
    is_connected : bool or None
        Whether the graph is connected.
    num_components : int or None
        Number of connected components.
    max_degree : int or None
        Maximum node degree.
    min_degree : int or None
        Minimum node degree.
    k : int or None
        Placeholder for k-subset analysis.
    total_k_subsets : int or None
        Placeholder for total k-subsets.
    pairable_subsets : int or None
        Placeholder for pairable subsets.
    pairable_ratio : float or None
        Placeholder for subset ratio.
    """
    graph_class: str | None = None
    attributes: list[str] = field(default_factory=list)
    nodes: int | None = None
    edges: int | None = None
    kind: str | None = None
    resource_type: str | None = None
    degree_sequence: list[int] = field(default_factory=list)
    spectrum: list[float] = field(default_factory=list)
    triangles: int | None = None
    is_connected: bool | None = None
    num_components: int | None = None
    max_degree: int | None = None
    min_degree: int | None = None
    k: int | None = None
    total_k_subsets: int | None = None
    pairable_subsets: int | None = None
    pairable_ratio: float | None = None


def analyze_resource_graph(resource_graph: GraphState) -> ResourceGraphInfo:
    """
    Analyze a GraphState object and extract metadata.

    Parameters
    ----------
    resource_graph : GraphState
        The graph state to be analyzed.

    Returns
    -------
    ResourceGraphInfo
        A dataclass object containing structural information.
    """
    graph_nx = nx.Graph(resource_graph.edges)

    return ResourceGraphInfo(
        graph_class=type(resource_graph).__name__,
        nodes=graph_nx.number_of_nodes(),
        edges=graph_nx.number_of_edges(),
        kind=getattr(resource_graph, "kind", None),
        resource_type=getattr(resource_graph, "type", None),
    )


class GraphStateExtractor:
    """
    Utility class for creating and analyzing graph states.
    """

    def __init__(self) -> None:
        """
        Initialize internal logs for extraction and equivalence timings.
        """
        self.extraction_times: list[float] = []
        self.equivalence_times: list[float] = []

    @staticmethod
    def create_2d_cluster_state(rows: int, cols: int) -> GraphState:
        """
        Create a rectangular 2D cluster state.

        Parameters
        ----------
        rows : int
            Number of rows in the grid.
        cols : int
            Number of columns in the grid.

        Returns
        -------
        GraphState
            A graph state representing the cluster lattice.
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
        Extract a subgraph (target graph state) from a larger cluster state.

        Parameters
        ----------
        cluster_state : GraphState
            The full 2D cluster state.
        target_edges : Sequence of tuple of int
            Edges defining the target subgraph.
        target_nodes : Sequence of int, optional
            Nodes to retain; inferred from edges if not provided.

        Returns
        -------
        tuple
            A tuple of:
            - GraphState: the extracted target graph.
            - list of int: the nodes measured (i.e., removed).
        """
        start = time.perf_counter()

        if target_nodes is None:
            target_nodes = list(set(itertools.chain.from_iterable(target_edges)))

        target_gs = GraphState()
        target_gs.add_nodes_from(target_nodes)
        target_gs.add_edges_from(target_edges)

        all_nodes = set(cluster_state.nodes)
        nodes_to_measure = list(all_nodes - set(target_nodes))

        self.extraction_times.append(time.perf_counter() - start)
        return target_gs, nodes_to_measure

    @staticmethod
    def compute_local_equivalence_invariants(gs: GraphState) -> ResourceGraphInfo:
        """
        Compute local graph invariants used to test local equivalence.

        Parameters
        ----------
        gs : GraphState
            Graph state whose invariants will be computed.

        Returns
        -------
        ResourceGraphInfo
            A dataclass object containing graph metrics like spectrum,
            connectivity, triangle count, and degree statistics.
        """
        graph = nx.Graph(gs.edges)

        degree_view = list(graph.degree)
        degree_sequence = sorted(int(d) for _, d in degree_view)

        return ResourceGraphInfo(
            nodes=graph.number_of_nodes(),
            edges=graph.number_of_edges(),
            degree_sequence=degree_sequence,
            spectrum=[float(x) for x in sorted(nx.adjacency_spectrum(graph).real)],
            triangles=sum(nx.triangles(graph).values()) // 3,
            is_connected=nx.is_connected(graph),
            num_components=nx.number_connected_components(graph),
            max_degree=max(degree_sequence) if degree_sequence else None,
            min_degree=min(degree_sequence) if degree_sequence else None,
        )


if __name__ == "__main__":
    extractor = GraphStateExtractor()
    cluster = extractor.create_2d_cluster_state(rows=3, cols=3)

    edges = [(0, 1), (1, 2)]
    target_graph, measured = extractor.extract_target_graph_state(cluster, edges)

    info = extractor.compute_local_equivalence_invariants(target_graph)

    print("\nGraph Analysis:")
    print("Nodes:", info.nodes)
    print("Edges:", info.edges)
    print("Degree sequence:", info.degree_sequence)
    print("Connected:", info.is_connected)
    print("Spectrum:", info.spectrum)
    print("Triangles:", info.triangles)
    print("Components:", info.num_components)
    print("Max degree:", info.max_degree)
    print("Min degree:", info.min_degree)
