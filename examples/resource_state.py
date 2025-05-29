"""Resource graph analysis and extraction tools."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import networkx as nx

from graphix import GraphState


@dataclass
class ResourceGraphInfo:
    """
    Information about a resource graph.
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


def analyze_resource_graph(resource_graph: object) -> ResourceGraphInfo:
    """
    Analyze a resource graph object and extract basic metadata.
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


class GraphStateExtractor:
    """
    Extract and analyze target graph states from cluster states.
    """

    def __init__(self) -> None:
        self.extraction_times: list[float] = []
        self.equivalence_times: list[float] = []

    @staticmethod
    def create_2d_cluster_state(rows: int, cols: int) -> GraphState:
        """
        Create a 2D cluster state.
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
        Compute invariants for local graph state equivalence.
        """
        info = ResourceGraphInfo()
        graph = nx.Graph(gs.edges)

        info.nodes = graph.number_of_nodes()
        info.edges = graph.number_of_edges()
        info.degree_sequence = sorted([d for _, d in graph.degree()])
        info.spectrum = sorted(nx.adjacency_spectrum(graph).real)
        info.triangles = sum(nx.triangles(graph).values()) // 3
        info.is_connected = nx.is_connected(graph)
        info.num_components = nx.number_connected_components(graph)
        info.max_degree = max(info.degree_sequence) if info.degree_sequence else None
        info.min_degree = min(info.degree_sequence) if info.degree_sequence else None

        return info
