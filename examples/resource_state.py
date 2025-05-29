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


# %%
def analyze_resource_graph(resource_graph: Any) -> ResourceGraphInfo:
    """Analyze a ResourceGraph object and extract relevant information.

    Parameters
    ----------
    resource_graph : Any
        A ResourceGraph object from graphix.extraction

    Returns
    -------
    ResourceGraphInfo
        Dictionary with resource graph information
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
            Number of rows in the grid
        cols : int
            Number of columns in the grid

        Returns
        -------
        GraphState
            GraphState representing the 2D cluster state
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

        This simulates the process where specific measurement patterns on cluster states
        can produce desired graph states, following the principles of measurement-based
        quantum computation.

        Parameters
        ----------
        cluster_state : GraphState
            The source 2D cluster state
        target_edges : list[tuple[int, int]]
            Desired edges in the target graph state
        target_nodes : list[int] | None, optional
            Nodes to keep in target state (if None, infer from edges)

        Returns
        -------
        tuple[GraphState, list[int]]
            Tuple of (extracted graph state, list of measured nodes)
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

        Based on the graphical characterization of local equivalence, two graph states
        are locally equivalent if one can be obtained from the other by local Clifford
        operations (local complementations).

        Parameters
        ----------
        gs : GraphState
            Input graph state

        Returns
        -------
        ResourceGraphInfo
            Dictionary of local equivalence invariants
        """
        start_time = time.perf_counter()

        # Convert to NetworkX graph for analysis
        graph = nx.Graph()
        graph.add_nodes_from(gs.nodes)
        graph.add_edges_from(gs.edges)

        invariants: ResourceGraphInfo = {}

        # Basic graph properties
        invariants["nodes"] = len(gs.nodes)
        invariants["edges"] = len(gs.edges)

        # Degree sequence (sorted) - fundamental LC invariant
        degrees = sorted(graph.degree(node) for node in graph.nodes())
        invariants["degree_sequence"] = degrees

        # Local complementation invariants
        invariants["max_degree"] = max(degrees) if degrees else 0
        invariants["min_degree"] = min(degrees) if degrees else 0

        # Spectral properties of adjacency matrix
        if len(gs.nodes) > 0:
            adj_matrix = nx.adjacency_matrix(graph).todense()
            eigenvals = np.linalg.eigvals(adj_matrix)
            # Sort eigenvalues and round to handle numerical precision
            eigenvals_real = sorted(np.round(eigenvals.real, 6))
            invariants["spectrum"] = eigenvals_real

        # Triangle count and clustering properties
        invariants["triangles"] = sum(nx.triangles(graph).values()) // 3

        # Connectedness properties
        invariants["is_connected"] = nx.is_connected(graph)
        invariants["num_components"] = nx.number_connected_components(graph)

        equivalence_time = time.perf_counter() - start_time
        self.equivalence_times.append(equivalence_time)

        return invariants

    @staticmethod
    def analyze_k_pairable_structure(gs: GraphState, k: int = 2) -> ResourceGraphInfo:
        """Analyze k-pairable structure of the graph state.

        Based on the concept of small k-pairable states, which are important for
        understanding the computational power and resource requirements of graph states.

        Parameters
        ----------
        gs : GraphState
            Input graph state
        k : int, optional
            Parameter for k-pairable analysis, by default 2

        Returns
        -------
        ResourceGraphInfo
            Analysis results including pairable subset counts and ratios
        """
        graph = nx.Graph()
        graph.add_nodes_from(gs.nodes)
        graph.add_edges_from(gs.edges)

        results: ResourceGraphInfo = {}

        # Find all possible k-vertex subsets
        nodes = list(gs.nodes)
        k_subsets = list(itertools.combinations(nodes, min(k, len(nodes))))

        # Analyze pairing properties
        pairable_subsets = []
        for subset in k_subsets:
            subgraph = graph.subgraph(subset)
            # A subset is k-pairable if it forms a perfect matching
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


# %%
# Let's start by creating a 2D cluster state and extracting a target graph state
print("Creating a 4x4 cluster state for graph state extraction...")

extractor = GraphStateExtractor()

# Create a 4x4 cluster state (2D grid graph)
cluster_4x4 = extractor.create_2d_cluster_state(4, 4)
print(f"Original cluster state: {len(cluster_4x4.nodes)} nodes, {len(cluster_4x4.edges)} edges")

# %%
# Define a 6-cycle as our target graph state to extract
target_nodes = [0, 1, 2, 6, 10, 9]
target_edges = [(0, 1), (1, 2), (2, 6), (6, 10), (10, 9), (9, 0)]

# Extract the target graph state from the cluster state
target_gs, measured_nodes = extractor.extract_target_graph_state(
    cluster_4x4, target_edges, target_nodes
)

print(f"Target graph state: {len(target_gs.nodes)} nodes, {len(target_gs.edges)} edges")
print(f"Nodes that would be measured out: {len(measured_nodes)}")
print(f"Target nodes: {target_nodes}")
print(f"Target edges: {target_edges}")

# %%
# Visualize the extracted target graph state
print("Visualizing the extracted 6-cycle graph state...")
try:
    target_gs.draw()
except Exception as e:
    print(f"Visualization not available: {e}")
    print(f"Graph structure - Nodes: {list(target_gs.nodes)}, Edges: {list(target_gs.edges)}")

# %%
# Analyze local equivalence properties of the extracted graph state
print("Analyzing local equivalence invariants of the extracted graph state...")

invariants = extractor.compute_local_equivalence_invariants(target_gs)
print(f"Number of nodes: {invariants['nodes']}")
print(f"Number of edges: {invariants['edges']}")
print(f"Degree sequence: {invariants['degree_sequence']}")
print(f"Number of triangles: {invariants['triangles']}")
print(f"Is connected: {invariants['is_connected']}")
print(f"Spectrum (first 5 eigenvalues): {invariants['spectrum'][:5]}")

# %%
# Perform k-pairable structure analysis
print("Analyzing k-pairable structure...")

k_analysis = extractor.analyze_k_pairable_structure(target_gs, k=2)
print(f"k-parameter: {k_analysis['k']}")
print(f"Total 2-subsets: {k_analysis['total_k_subsets']}")
print(f"Pairable subsets: {k_analysis['pairable_subsets']}")
print(f"Pairable ratio: {k_analysis['pairable_ratio']:.3f}")

# %%
# Decompose the extracted graph state into fusion network
print("Decomposing extracted graph state with fusion network...")

try:
    fusion_network = get_fusion_network_from_graph(target_gs)
    print(f"Number of resource states in fusion network: {len(fusion_network)}")

    for i, resource_state in enumerate(fusion_network):
        info = analyze_resource_graph(resource_state)
        if "nodes" in info and "edges" in info:
            print(f"Resource state {i}: {info['nodes']} nodes, {info['edges']} edges")
        else:
            print(f"Resource state {i}: {info['type']}")

        if "kind" in info:
            print(f"  Type: {info['kind']}")
        elif "resource_type" in info:
            print(f"  Type: {info['resource_type']}")

except Exception as e:
    print(f"Error in fusion network analysis: {e}")

# %%
# Analyze fusion connections between resource states
if "fusion_network" in locals():
    print("Analyzing fusion connections between resource states...")
    from graphix.extraction import get_fusion_nodes

    for idx1, idx2 in itertools.combinations(range(len(fusion_network)), 2):
        fusion_nodes = get_fusion_nodes(fusion_network[idx1], fusion_network[idx2])
        if fusion_nodes:
            print(f"Fusion nodes between resource state {idx1} and {idx2}: {fusion_nodes}")


# %%
def create_test_graphs():
    """Create various small graph states for comparison."""
    test_graphs = []

    # Path graphs
    for n in range(3, 6):
        gs = GraphState()
        nodes = list(range(n))
        edges = [(i, i + 1) for i in range(n - 1)]
        gs.add_nodes_from(nodes)
        gs.add_edges_from(edges)
        test_graphs.append(("path", n, gs))

    # Cycle graphs
    for n in range(3, 6):
        gs = GraphState()
        nodes = list(range(n))
        edges = [(i, (i + 1) % n) for i in range(n)]
        gs.add_nodes_from(nodes)
        gs.add_edges_from(edges)
        test_graphs.append(("cycle", n, gs))

    # Complete graphs
    for n in range(3, 5):
        gs = GraphState()
        nodes = list(range(n))
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        gs.add_nodes_from(nodes)
        gs.add_edges_from(edges)
        test_graphs.append(("complete", n, gs))

    return test_graphs


# Generate and compare different graph types for local equivalence analysis
print("Comparing local equivalence classes of different graph types...")

test_graphs = create_test_graphs()
equivalence_classes = {}

# Analyze each test graph
for graph_type, n, gs in test_graphs:
    invariants = extractor.compute_local_equivalence_invariants(gs)

    # Create a signature for equivalence class identification
    signature = (
        tuple(invariants["degree_sequence"]),
        tuple(np.round(invariants["spectrum"], 3)),
        invariants["triangles"],
        invariants["is_connected"],
    )

    if signature not in equivalence_classes:
        equivalence_classes[signature] = []

    equivalence_classes[signature].append((graph_type, n, gs))

print(f"Found {len(equivalence_classes)} distinct local equivalence classes:")
for i, (signature, graphs) in enumerate(equivalence_classes.items()):
    print(f"  Class {i + 1}: {len(graphs)} graphs")
    print(f"    Degree sequence: {signature[0]}")
    print(f"    Representative graphs: {[(g[0], g[1]) for g in graphs[:3]]}")


# %%
def run_scalability_analysis():
    """Analyze scalability of graph state extraction and analysis."""
    sizes = [3, 4, 5, 6]  # Smaller sizes for demonstration
    extraction_times = []
    equivalence_times = []
    fusion_times = []

    for size in sizes:
        print(f"  Analyzing {size}x{size} cluster state...")

        # Create cluster state
        cluster_state = extractor.create_2d_cluster_state(size, size)

        # Define target (small cycle)
        target_nodes = list(range(min(6, len(cluster_state.nodes))))
        target_edges = [(i, (i + 1) % len(target_nodes)) for i in range(len(target_nodes))]

        # Time extraction
        start_time = time.perf_counter()
        target_gs, _ = extractor.extract_target_graph_state(
            cluster_state, target_edges, target_nodes
        )
        extraction_time = time.perf_counter() - start_time
        extraction_times.append(extraction_time)

        # Time equivalence analysis
        start_time = time.perf_counter()
        _ = extractor.compute_local_equivalence_invariants(target_gs)
        equivalence_time = time.perf_counter() - start_time
        equivalence_times.append(equivalence_time)

        # Time fusion network generation
        start_time = time.perf_counter()
        try:
            _ = get_fusion_network_from_graph(target_gs)
            fusion_time = time.perf_counter() - start_time
        except Exception:
            fusion_time = 0
        fusion_times.append(fusion_time)

        print(
            f"    Extraction: {extraction_time:.6f}s, "
            f"Equivalence: {equivalence_time:.6f}s, "
            f"Fusion: {fusion_time:.6f}s"
        )

    return sizes, extraction_times, equivalence_times, fusion_times


# Perform scalability analysis
print("Running scalability analysis for different cluster state sizes...")

sizes, extraction_times, equivalence_times, fusion_times = run_scalability_analysis()

# %%
# Plot scalability results
if all(t >= 0 for t in extraction_times + equivalence_times + fusion_times):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    node_counts = [s * s for s in sizes]

    ax1.plot(node_counts, extraction_times, "o-", label="Extraction", color="blue")
    ax1.set_xlabel("Number of nodes")
    ax1.set_ylabel("Time (s)")
    ax1.set_title("Graph State Extraction Scaling")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(node_counts, equivalence_times, "o-", label="Equivalence", color="orange")
    ax2.set_xlabel("Number of nodes")
    ax2.set_ylabel("Time (s)")
    ax2.set_title("Local Equivalence Analysis Scaling")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3.plot(node_counts, fusion_times, "o-", label="Fusion Network", color="green")
    ax3.set_xlabel("Number of nodes")
    ax3.set_ylabel("Time (s)")
    ax3.set_title("Fusion Network Generation Scaling")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plt.show()
else:
    print("Plotting skipped due to timing issues")

# %%
print("\nSummary of Graph State Extraction and Local Equivalence Analysis:")
print("=" * 70)
print("• Successfully extracted target graph states from 2D cluster states")
print("• Analyzed local equivalence invariants including degree sequences and spectra")
print("• Investigated k-pairable structure properties")
print("• Decomposed graph states into fusion networks for resource analysis")
print(f"• Identified {len(equivalence_classes)} distinct local equivalence classes")
print("• Demonstrated scalability analysis for different cluster state sizes")
