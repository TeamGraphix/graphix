"""
Graph State Resource Extraction
===============================

This module demonstrates how to extract and analyze a resource graph from a
2D cluster state using the GraphState class. It includes functionality for
computing graph invariants, analyzing connectivity, and comparing local
equivalence between graph states.

Classes
-------
ResourceGraphInfo
    Data structure to store metadata about a graph state.

GraphStateExtractor
    Utility class to create, extract, and analyze graph states.

Functions
---------
analyze_resource_graph(resource_graph)
    Extract basic metadata from a resource graph object.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
import networkx as nx

from graphix import GraphState


@dataclass
class ResourceGraphInfo:
    """
    Information container for metadata about a resource graph.

    Attributes
    ----------
    type : str or None
        String representation of the graph object type.
    attributes : list of str
        List of attribute names (if applicable).
    nodes : int or None
        Total number of nodes in the graph.
    edges : int or None
        Total number of edges in the graph.
    kind : str or None
        Graph kind, if defined by the object.
    resource_type : str or None
        Type information for custom graph resource.
    degree_sequence : list of int
        Degree of each node in sorted order.
    spectrum : list of float
        Real parts of the adjacency matrix spectrum.
    triangles : int or None
        Number of triangles in the graph.
    is_connected : bool or None
        Whether the graph is fully connected.
    num_components : int or None
        Number of connected components in the graph.
    max_degree : int or None
        Maximum degree of any node.
    min_degree : int or None
        Minimum degree of any node.
    k : int or None
        Size of subsets (for advanced analysis).
    total_k_subsets : int or None
        Total number of k-subsets (if used).
    pairable_subsets : int or None
        Number of k-subsets that are pairable.
    pairable_ratio : float or None
        Fraction of pairable subsets.
    """


def analyze_resource_graph(resource_graph: GraphState) -> ResourceGraphInfo:
    """
    Analyze a GraphState object and extract structural metadata.

    Parameters
    ----------
    resource_graph : GraphState
        A graph-like object with node and edge attributes.

    Returns
    -------
    ResourceGraphInfo
        A dataclass instance containing structural information about the graph.
    """
    info = ResourceGraphInfo(type=type(resource_graph).__name__)

    graph = getattr(resource_graph, "graph", None)
    if graph is not None:
        info.nodes = len(getattr(graph, "nodes", []))
        info.edges = len(getattr(graph, "edges", []))
    elif hasattr(resource_graph, "nodes") and hasattr(resource_graph, "edges"):
        info.nodes = len(resource_graph.nodes)
        info.edges = len(resource_graph.edges)
    elif hasattr(resource_graph, "n_node"):
        info.nodes = resource_graph.n_node

    if hasattr(resource_graph, "kind"):
        info.kind = resource_graph.kind
    elif hasattr(resource_graph, "type"):
        info.resource_type = resource_graph.type

    return info


class GraphStateExtractor:
    """
    Extract and analyze target graph states from cluster states.

    Methods
    -------
    create_2d_cluster_state(rows, cols)
        Construct a rectangular cluster state as a graph.

    extract_target_graph_state(cluster_state, target_edges, target_nodes)
        Subset a cluster graph into a smaller target graph state.

    compute_local_equivalence_invariants(gs)
        Compute degree spectrum, connectivity, and triangle counts.
    """

    def __init__(self) -> None:
        self.extraction_times: list[float] = []
        self.equivalence_times: list[float] = []

    @staticmethod
    def create_2d_cluster_state(rows: int, cols: int) -> GraphState:
        """
        Create a 2D cluster state graph.

        Parameters
        ----------
        rows : int
            Number of rows in the cluster state grid.
        cols : int
            Number of columns in the cluster state grid.

        Returns
        -------
        GraphState
            A 2D grid cluster state encoded as a graph.
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
        Extract a target graph state using local measurements.

        Parameters
        ----------
        cluster_state : GraphState
            The original cluster graph.
        target_edges : list of tuple of int
            Edges in the extracted target graph.
        target_nodes : list of int, optional
            Node list for extraction. If None, inferred from edges.

        Returns
        -------
        tuple
            A tuple containing:
            - GraphState: the extracted subgraph
            - list[int]: list of measured-away nodes
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
        Compute invariants useful for checking local unitary equivalence.

        Parameters
        ----------
        gs : GraphState
            A graph state object to analyze.

        Returns
        -------
        ResourceGraphInfo
            Metadata including degree sequence, adjacency spectrum,
            connectivity, and triangle count.
        """
        info = ResourceGraphInfo()
        graph = nx.Graph(gs.edges)

        info.nodes = graph.number_of_nodes()
        info.edges = graph.number_of_edges()

        degree_view = list(graph.degree)
        info.degree_sequence = sorted(int(d) for _, d in degree_view)

        info.spectrum = [float(x) for x in sorted(nx.adjacency_spectrum(graph).real)]
        info.triangles = sum(nx.triangles(graph).values()) // 3
        info.is_connected = nx.is_connected(graph)
        info.num_components = nx.number_connected_components(graph)
        info.max_degree = max(info.degree_sequence) if info.degree_sequence else None
        info.min_degree = min(info.degree_sequence) if info.degree_sequence else None

        return info
