"""Script for creating and analyzing graph states using Graphix.

Includes:
- Cluster state generation
- Target graph state extraction
- Local equivalence analysis
- k-pairable structure analysis
- Fusion network decomposition
"""

# %%
from __future__ import annotations

import itertools
import time
from typing import TYPE_CHECKING, Any, TypedDict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from graphix import GraphState
from graphix.extraction import get_fusion_network_from_graph

if TYPE_CHECKING:
    from typing import Optional


# %%
class ResourceGraphInfo(TypedDict, total=False):
    """Type definition for resource graph information."""

    type: str
    attributes: list[str]
    nodes: int
    edges: int
    kind: str
    resource_type: str
    degree_sequence: list[int]
    spectrum: list[float]
    triangles: int
    is_connected: bool
    num_components: int
    max_degree: int
    min_degree: int
    k: int
    total_k_subsets: int
    pairable_subsets: int
    pairable_ratio: float


# %%
def analyze_resource_graph(resource_graph: Any) -> ResourceGraphInfo:
    """Analyze a ResourceGraph object and extract relevant information.

    Parameters
    ----------
    resource_graph : Any
        A ResourceGraph object from graphix.extraction.

    Returns
    -------
    ResourceGraphInfo
        Dictionary with resource graph information.
    """
    info: ResourceGraphInfo = {
        "type": type(resource_graph).__name__,
    }

    # Try to get size information
    if hasattr(resource_graph, "graph"):
        graph = resource_graph.graph
        if hasattr(graph, "nodes") and hasattr(graph, "edges"):
            info["nodes"] = len(graph.nodes)
            info["edges"] = len(graph.edges)
    elif hasattr(resource_graph, "nodes") and hasattr(resource_graph, "edges"):
        info["nodes"] = len(resource_graph.nodes)
        info["edges"] = len(resource_graph.edges)
    elif hasattr(resource_graph, "n_node"):
        info["nodes"] = resource_graph.n_node

    # Try to get resource type information
    if hasattr(resource_graph, "kind"):
        info["kind"] = resource_graph.kind
    elif hasattr(resource_graph, "type"):
        info["resource_type"] = resource_graph.type

    return info


# %%
class GraphStateExtractor:
    """Extract target graph states from 2D cluster states and analyze local equivalence."""

    def __init__(self) -> None:
        """Initialize the extractor with empty timing lists."""
        self.extraction_times = []
        self.equivalence_times = []

    @staticmethod
    def create_2d_cluster_state(rows: int, cols: int) -> GraphState:
        """Create a 2D cluster state (grid graph) with given dimensions.

        Parameters
        ----------
        rows : int
            Number of rows in the grid.
        cols : int
            Number of columns in the grid.

        Returns
        -------
        GraphState
            GraphState representing the 2D cluster state.
        """
        gs = GraphState()

        # Add all nodes
        nodes = []
        for i in range(rows):
            for j in range(cols):
                node_id = i * cols + j
                nodes.append(node_id)

        gs.add_nodes_from(nodes)

        # Add edges for 2D grid connectivity
        edges = []
        for i in range(rows):
            for j in range(cols):
                current = i * cols + j

                # Connect to right neighbor
                if j < cols - 1:
                    right = i * cols + (j + 1)
                    edges.append((current, right))

                # Connect to bottom neighbor
                if i < rows - 1:
                    bottom = (i + 1) * cols + j
                    edges.append((current, bottom))

        gs.add_edges_from(edges)
        return gs

    def extract_target_graph_state(
        self,
        cluster_state: GraphState,
        target_edges: list[tuple[int, int]],
        target_nodes: list[int] | None = None,
    ) -> tuple[GraphState, list[int]]:
        """Extract a target graph state from a 2D cluster state using local measurements.

        Parameters
        ----------
        cluster_state : GraphState
            The source 2D cluster state.
        target_edges : list[tuple[int, int]]
            Desired edges in the target graph state.
        target_nodes : list[int] | None, optional
            Nodes to keep in target state (if None, infer from edges).

        Returns
        -------
        tuple[GraphState, list[int]]
            Tuple of (extracted graph state, list of measured nodes).
        """
        start_time = time.perf_counter()

        if target_nodes is None:
            target_nodes = list(set(itertools.chain(*target_edges)))

        # Create the target graph state
        target_gs = GraphState()
        target_gs.add_nodes_from(target_nodes)
        target_gs.add_edges_from(target_edges)

        # Determine which nodes need to be measured out
        all_cluster_nodes = set(cluster_state.nodes)
        nodes_to_measure = all_cluster_nodes - set(target_nodes)

        extraction_time = time.perf_counter() - start_time
        self.extraction_times.append(extraction_time)

        return target_gs, list(nodes_to_measure)

    def compute_local_equivalence_invariants(self, gs: GraphState) -> ResourceGraphInfo:
        """Compute invariants that characterize the local equivalence class of a graph state.

        Parameters
        ----------
        gs : GraphState
            Input graph state.

        Returns
        -------
        ResourceGraphInfo
            Dictionary of local equivalence invariants.
        """
        start_time = time.perf_counter()

        graph = nx.Graph()
        graph.add_nodes_from(gs.nodes)
        graph.add_edges_from(gs.edges)

        invariants: ResourceGraphInfo = {}

        invariants["nodes"] = len(gs.nodes)
        invariants["edges"] = len(gs.edges)

        degrees = sorted(graph.degree(node) for node in graph.nodes())
        invariants["degree_sequence"] = degrees
        invariants["max_degree"] = max(degrees) if degrees else 0
        invariants["min_degree"] = min(degrees) if degrees else 0

        if len(gs.nodes) > 0:
            adj_matrix = nx.adjacency_matrix(graph).todense()
            eigenvals = np.linalg.eigvals(adj_matrix)
            eigenvals_real = sorted(np.round(eigenvals.real, 6))
            invariants["spectrum"] = eigenvals_real

        invariants["triangles"] = sum(nx.triangles(graph).values()) // 3
        invariants["is_connected"] = nx.is_connected(graph)
        invariants["num_components"] = nx.number_connected_components(graph)

        equivalence_time = time.perf_counter() - start_time
        self.equivalence_times.append(equivalence_time)

        return invariants

    @staticmethod
    def analyze_k_pairable_structure(gs: GraphState, k: int = 2) -> ResourceGraphInfo:
        """Analyze k-pairable structure of the graph state.

        Parameters
        ----------
        gs : GraphState
            Input graph state.
        k : int, optional
            Parameter for k-pairable analysis, by default 2.

        Returns
        -------
        ResourceGraphInfo
            Analysis results including pairable subset counts and ratios.
        """
        graph = nx.Graph()
        graph.add_nodes_from(gs.nodes)
        graph.add_edges_from(gs.edges)

        results: ResourceGraphInfo = {}

        nodes = list(gs.nodes)
        k_subsets = list(itertools.combinations(nodes, min(k, len(nodes))))
        pairable_subsets = []

        for subset in k_subsets:
            subgraph = graph.subgraph(subset)
            if len(subset) % 2 == 0:
                matching = nx.max_weight_matching(subgraph)
                if nx.is_perfect_matching(subgraph, matching):
                    pairable_subsets.append(subset)

        results["k"] = k
        results["total_k_subsets"] = len(k_subsets)
        results["pairable_subsets"] = len(pairable_subsets)
        results["pairable_ratio"] = (
            len(pairable_subsets) / len(k_subsets) if k_subsets else 0
        )

        return results

