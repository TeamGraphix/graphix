"""
Graph State Resource Extraction
===============================

This module demonstrates how to extract and analyze a resource graph from a
2D cluster state using the GraphState class from the `graphix` library.

It includes analysis of graph invariants, connectivity, and local equivalence
using NetworkX.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import networkx as nx
from graphix import GraphState


@dataclass
class ResourceGraphInfo:
    """Container for resource graph metrics.

    Attributes
    ----------
    type : str or None
        The class name of the graph-like object.
    attributes : list of str
        Any string-based attributes to record (optional).
    nodes : int or None
        Number of nodes in the graph.
    edges : int or None
        Number of edges in the graph.
    kind : str or None
        Additional kind/type info (optional).
    resource_type : str or None
        Explicit label of resource type, if available.
    degree_sequence : list of int
        Sorted list of node degrees.
    spectrum : list of float
        Real parts of eigenvalues of the adjacency matrix.
    triangles : int or None
        Number of triangles in the graph.
    is_connected : bool or None
        Whether the graph is fully connected.
    num_components : int or None
        Number of connected components.
    max_degree : int or None
        Maximum node degree.
    min_degree : int or None
        Minimum node degree.
    k : int or None
        Optional parameter for combinatorial analysis.
    total_k_subsets : int or None
        Number of subsets of size k (if analyzed).
    pairable_subsets : int or None
        Number of pairable k-subsets (if applicable).
    pairable_ratio : float or None
        Ratio of pairable subsets to total.
    """

    type: str | None = None
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
    Analyze a GraphState object and return metadata.

    Parameters
    ----------
    resource_graph : GraphState
        The input graph state to analyze.

    Returns
    -------
    ResourceGraphInfo
        Metadata including node/edge counts, type, and structure.
    """
    return ResourceGraphInfo(
        type=type(resource_graph).__name__,
        nodes=len(resource_graph.graph.nodes),
        edges=len(resource_graph.graph.edges),
        kind=resource_graph.kind,
        resource_type=resource_graph.type,
    )


class GraphStateExtractor:
    """
    Utility class for creating and analyzing graph states.
    """

    def __init__(self) -> None:
        """Initialize timing logs."""
        self.extraction_times: list[float] = []
        self.equivalence_times: list[float] = []

    @staticmethod
    def create_2d_cluster_state(rows: int, cols: int) -> GraphState:
        """
        Create a 2D cluster state as a rectangular lattice.

        Parameters
        ----------
        rows : int
            Number of rows in the grid.
        cols : int
            Number of columns in the grid.

        Returns
        -------
        GraphState
            The constructed 2D cluster graph.
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
        target_edges: list[tuple[int, int]],
        target_nodes: list[int] | None = None,
    ) -> tuple[GraphState, list[int]]:
        """
        Extract a graph state using local measurements.

        Parameters
        ----------
        cluster_state : GraphState
            The source 2D cluster graph.
        target_edges : list of tuple of int
            Edges to preserve in the target graph.
        target_nodes : list of int or None
            Nodes to preserve. If None, inferred from edges.

        Returns
        -------
        tuple of (GraphState, list of int)
            Extracted target graph state and nodes to be measured.
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
        Compute local invariants of a graph state for equivalence testing.

        Parameters
        ----------
        gs : GraphState
            The graph state to analyze.

        Returns
        -------
        ResourceGraphInfo
            Information about degree distribution, triangle count, etc.
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
