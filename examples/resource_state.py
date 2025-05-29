# %%
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from graphix import GraphState
from graphix.extraction import get_fusion_network_from_graph


# %%
@dataclass
class ResourceGraphInfo:
    """
    Data class for resource graph information.
    """
    type: Optional[str] = None
    attributes: list[str] = field(default_factory=list)
    nodes: Optional[int] = None
    edges: Optional[int] = None
    kind: Optional[str] = None
    resource_type: Optional[str] = None
    degree_sequence: list[int] = field(default_factory=list)
    spectrum: list[float] = field(default_factory=list)
    triangles: Optional[int] = None
    is_connected: Optional[bool] = None
    num_components: Optional[int] = None
    max_degree: Optional[int] = None
    min_degree: Optional[int] = None
    k: Optional[int] = None
    total_k_subsets: Optional[int] = None
    pairable_subsets: Optional[int] = None
    pairable_ratio: Optional[float] = None


# %%
def analyze_resource_graph(resource_graph: object) -> ResourceGraphInfo:
    """
    Analyze a resource graph object and extract basic information.

    Parameters
    ----------
    resource_graph : object
        A ResourceGraph object from `graphix.extraction`.

    Returns
    -------
    ResourceGraphInfo
        Extracted graph info.
    """
    info = ResourceGraphInfo(type=type(resource_graph).__name__)

    if hasattr(resource_graph, "graph"):
        graph = resource_graph.graph
        info.nodes = len(graph.nodes) if hasattr(graph, "nodes") else None
        info.edges = len(graph.edges) if hasattr(graph, "edges") else None
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


# %%
class GraphStateExtractor:
    """
    Extract and analyze target graph states from cluster states.
    """

    def __init__(self) -> None:
        """
        Initialize the extractor with timing records.
        """
        self.extraction_times: list[float] = []
        self.equivalence_times: list[float] = []

    @staticmethod
    def create_2d_cluster_state(rows: int, cols: int) -> GraphState:
        """
        Create a 2D cluster state.

        Parameters
        ----------
        rows : int
            Number of rows.
        cols : int
            Number of columns.

        Returns
        -------
        GraphState
            Generated cluster state.
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
        target_nodes: Optional[list[int]] = None,
    ) -> tuple[GraphState, list[int]]:
        """
        Extract a target graph state using local measurements.

        Parameters
        ----------
        cluster_state : GraphState
            Source 2D cluster state.
        target_edges : list[tuple[int, int]]
            Edges to preserve.
        target_nodes : list[int], optional
            Nodes to retain.

        Returns
        -------
        tuple[GraphState, list[int]]
            Extracted graph and list of measured-out nodes.
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

    def compute_local_equivalence_invariants(self, gs: GraphState) -> ResourceGraphInfo:
        """
        Compute local equivalence invariants of a graph state.

        Parameters
        ----------
        gs : GraphState

        Returns
        -------
        ResourceGraphInfo
        """
        start = time.perf_counter()

        graph = nx.Graph()
        graph.add_nodes_from(gs.nodes)
        graph.add_edges_from(gs.edges)

        info = ResourceGraphInfo(
            nodes=len(gs.nodes),
            edges=len(gs.edges),
            degree_sequence=sorted(dict(graph.degree()).values()),
            max_degree=max(dict(graph.degree()).values(), default=0),
            min_degree=min(dict(graph.degree()).values(), default=0),
            triangles=sum(nx.triangles(graph).values()) // 3,
            is_connected=nx.is_connected(graph),
            num_components=nx.number_connected_components(graph),
        )

        if gs.nodes:
            adj = nx.adjacency_matrix(graph).todense()
            eigenvals = np.linalg.eigvals(adj)
            info.spectrum = sorted(np.round(eigenvals.real, 6))

        self.equivalence_times.append(time.perf_counter() - start)
        return info

    @staticmethod
    def analyze_k_pairable_structure(gs: GraphState, k: int = 2) -> ResourceGraphInfo:
        """
        Analyze k-pairable structure of the graph.

        Parameters
        ----------
        gs : GraphState
        k : int

        Returns
        -------
        ResourceGraphInfo
        """
        graph = nx.Graph()
        graph.add_nodes_from(gs.nodes)
        graph.add_edges_from(gs.edges)

        nodes = list(gs.nodes)
        k_subsets = list(itertools.combinations(nodes, min(k, len(nodes))))
        pairable_subsets = []

        for subset in k_subsets:
            sub = graph.subgraph(subset)
            if len(subset) % 2 == 0:
                matching = nx.max_weight_matching(sub)
                if nx.is_perfect_matching(sub, matching):
                    pairable_subsets.append(subset)

        return ResourceGraphInfo(
            k=k,
            total_k_subsets=len(k_subsets),
            pairable_subsets=len(pairable_subsets),
            pairable_ratio=len(pairable_subsets) / len(k_subsets) if k_subsets else 0.0,
        )
