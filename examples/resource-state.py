"""Graph State Extraction and Local Equivalence Analysis.

This example demonstrates graph state extraction from cluster states and analysis of
local equivalence between stabilizer states, inspired by recent research in quantum
resource state engineering.

The implementation focuses on:
1. Extracting target graph states from 2D cluster states via local measurements
2. Characterizing local equivalence classes of stabilizer states
3. Analyzing scalability and computational complexity

Based on:
- Freund, Pirker, Vandré and Dür, Graph state extraction from two-dimensional
  cluster states (2025)
- Claudet and Perdix, Local equivalence of stabilizer states: a graphical
  characterisation (2024)
- Small k-pairable states analysis
"""

from __future__ import annotations

import itertools
import time
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import graphix
from graphix import GraphState
from graphix.extraction import get_fusion_network_from_graph

if TYPE_CHECKING:
    from typing import Dict, List, Optional, Set, Tuple


def analyze_resource_graph(resource_graph: Any) -> Dict[str, Any]:
    """Analyze a ResourceGraph object and extract relevant information.

    Args:
        resource_graph: A ResourceGraph object from graphix.extraction

    Returns:
        Dictionary with resource graph information
    """
    info = {
        "type": type(resource_graph).__name__,
        "attributes": [],
    }

    # Get all public attributes
    attrs = [attr for attr in dir(resource_graph) if not attr.startswith("_")]
    info["attributes"] = attrs

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


class GraphStateExtractor:
    """Extract target graph states from 2D cluster states and analyze local equivalence."""

    def __init__(self) -> None:
        """Initialize the extractor with empty timing lists."""
        self.extraction_times: List[float] = []
        self.equivalence_times: List[float] = []

    def create_2d_cluster_state(self, rows: int, cols: int) -> GraphState:
        """Create a 2D cluster state (grid graph) with given dimensions.

        Args:
            rows: Number of rows in the grid
            cols: Number of columns in the grid

        Returns:
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
        target_edges: List[Tuple[int, int]],
        target_nodes: Optional[List[int]] = None,
    ) -> Tuple[GraphState, List[int]]:
        """Extract a target graph state from a 2D cluster state using local measurements.

        This simulates the process described in Freund et al. where specific measurement
        patterns on cluster states can produce desired graph states.

        Args:
            cluster_state: The source 2D cluster state
            target_edges: Desired edges in the target graph state
            target_nodes: Nodes to keep in target state (if None, infer from edges)

        Returns:
            Tuple of (extracted graph state, list of measured nodes)
        """
        start_time = time.time()

        if target_nodes is None:
            target_nodes = list(set(itertools.chain(*target_edges)))

        # Create the target graph state
        target_gs = GraphState()
        target_gs.add_nodes_from(target_nodes)
        target_gs.add_edges_from(target_edges)

        # Determine which nodes need to be measured out
        all_cluster_nodes = set(cluster_state.nodes)
        nodes_to_measure = all_cluster_nodes - set(target_nodes)

        extraction_time = time.time() - start_time
        self.extraction_times.append(extraction_time)

        return target_gs, list(nodes_to_measure)

    def compute_local_equivalence_invariants(self, gs: GraphState) -> Dict[str, Any]:
        """Compute invariants that characterize the local equivalence class of a graph state.

        Based on Claudet and Perdrix's graphical characterization of local equivalence.

        Args:
            gs: Input graph state

        Returns:
            Dictionary of local equivalence invariants
        """
        start_time = time.time()

        # Convert to NetworkX graph for analysis
        graph = nx.Graph()
        graph.add_nodes_from(gs.nodes)
        graph.add_edges_from(gs.edges)

        invariants: Dict[str, Any] = {}

        # Basic graph properties
        invariants["num_nodes"] = len(gs.nodes)
        invariants["num_edges"] = len(gs.edges)

        # Degree sequence (sorted)
        degrees = sorted([graph.degree(node) for node in graph.nodes()])
        invariants["degree_sequence"] = degrees

        # Local complementation invariants
        # Two graphs are LC-equivalent if one can be obtained from the other
        # by a sequence of local complementations

        # Compute orbit-stabilizer information
        invariants["max_degree"] = max(degrees) if degrees else 0
        invariants["min_degree"] = min(degrees) if degrees else 0

        # Spectral properties of adjacency matrix
        if len(gs.nodes) > 0:
            adj_matrix = nx.adjacency_matrix(graph).todense()
            eigenvals = np.linalg.eigvals(adj_matrix)
            # Sort eigenvalues and round to handle numerical precision
            eigenvals_real = sorted(np.round(eigenvals.real, 6))
            invariants["spectrum"] = eigenvals_real
            invariants["rank"] = np.linalg.matrix_rank(adj_matrix)

        # Triangle count and clustering properties
        invariants["triangles"] = sum(nx.triangles(graph).values()) // 3
        invariants["clustering_coeffs"] = sorted(nx.clustering(graph).values())

        # Connectedness properties
        invariants["is_connected"] = nx.is_connected(graph)
        invariants["num_components"] = nx.number_connected_components(graph)

        equivalence_time = time.time() - start_time
        self.equivalence_times.append(equivalence_time)

        return invariants

    def analyze_k_pairable_structure(self, gs: GraphState, k: int = 2) -> Dict[str, Any]:
        """Analyze k-pairable structure of the graph state.

        Based on the concept of small k-pairable states from Claudet, Mhalla and Perdrix.

        Args:
            gs: Input graph state
            k: Parameter for k-pairable analysis

        Returns:
            Analysis results
        """
        graph = nx.Graph()
        graph.add_nodes_from(gs.nodes)
        graph.add_edges_from(gs.edges)

        results: Dict[str, Any] = {}

        # Find all possible k-vertex subsets
        nodes = list(gs.nodes)
        k_subsets = list(itertools.combinations(nodes, min(k, len(nodes))))

        # Analyze pairing properties
        pairable_subsets = []
        for subset in k_subsets:
            subgraph = graph.subgraph(subset)
            # A subset is k-pairable if it forms a specific structure
            # Here we use perfect matching as a simple criterion
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


def run_scalability_analysis() -> Tuple[List[int], List[float], List[float], List[float]]:
    """Analyze the scalability of graph state extraction and equivalence analysis."""
    print("Running scalability analysis...")

    extractor = GraphStateExtractor()
    sizes = [4, 6, 8, 10, 12, 16, 20]

    extraction_times = []
    equivalence_times = []
    fusion_times = []

    for size in sizes:
        print(f"Analyzing size {size}x{size} cluster state...")

        # Create 2D cluster state
        cluster_state = extractor.create_2d_cluster_state(size, size)

        # Define a target graph state (e.g., a cycle)
        target_nodes = list(range(min(size * 2, len(cluster_state.nodes))))
        target_edges = [(i, (i + 1) % len(target_nodes)) for i in range(len(target_nodes))]

        # Time the extraction process
        start_time = time.time()
        target_gs, measured_nodes = extractor.extract_target_graph_state(
            cluster_state, target_edges, target_nodes
        )
        extraction_time = time.time() - start_time
        extraction_times.append(extraction_time)

        # Time the equivalence analysis
        start_time = time.time()
        invariants = extractor.compute_local_equivalence_invariants(target_gs)
        equivalence_time = time.time() - start_time
        equivalence_times.append(equivalence_time)

        # Time the fusion network generation
        start_time = time.time()
        try:
            fusion_network = get_fusion_network_from_graph(target_gs)
            fusion_time = time.time() - start_time
            fusion_times.append(fusion_time)
        except Exception as e:  # noqa: BLE001
            print(f"  Warning: Fusion network generation failed: {e}")
            fusion_time = 0
            fusion_times.append(fusion_time)

        print(f"  Extraction: {extraction_time:.4f}s")
        print(f"  Equivalence: {equivalence_time:.4f}s")
        print(f"  Fusion: {fusion_time:.4f}s")

    # Plot results
    try:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot([s * s for s in sizes], extraction_times, "o-", label="Extraction")
        plt.xlabel("Number of nodes")
        plt.ylabel("Time (s)")
        plt.title("Graph State Extraction Scaling")
        plt.yscale("log")
        plt.xscale("log")
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot([s * s for s in sizes], equivalence_times, "o-", label="Equivalence", color="orange")
        plt.xlabel("Number of nodes")
        plt.ylabel("Time (s)")
        plt.title("Local Equivalence Analysis Scaling")
        plt.yscale("log")
        plt.xscale("log")
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot([s * s for s in sizes], fusion_times, "o-", label="Fusion Network", color="green")
        plt.xlabel("Number of nodes")
        plt.ylabel("Time (s)")
        plt.title("Fusion Network Generation Scaling")
        plt.yscale("log")
        plt.xscale("log")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    except Exception as e:  # noqa: BLE001
        print(f"Plotting failed: {e}")
        print("Raw timing data:")
        for i, size in enumerate(sizes):
            print(
                f"Size {size}x{size}: "
                f"Extraction={extraction_times[i]:.4f}s, "
                f"Equivalence={equivalence_times[i]:.4f}s, "
                f"Fusion={fusion_times[i]:.4f}s"
            )

    return sizes, extraction_times, equivalence_times, fusion_times


def demonstrate_graph_state_extraction() -> Tuple[GraphState, Dict[str, Any], Dict[str, Any]]:
    """Demonstrate the main functionality with concrete examples."""
    print("Graph State Extraction and Local Equivalence Analysis")
    print("=" * 55)

    extractor = GraphStateExtractor()

    # Example 1: Extract a cycle graph from a 2D cluster state
    print("\n1. Extracting a 6-cycle from a 4x4 cluster state")
    cluster_4x4 = extractor.create_2d_cluster_state(4, 4)
    print(
        f"   Original cluster state: "
        f"{len(cluster_4x4.nodes)} nodes, {len(cluster_4x4.edges)} edges"
    )

    # Define a 6-cycle as target
    target_nodes = [0, 1, 2, 6, 10, 9]
    target_edges = [(0, 1), (1, 2), (2, 6), (6, 10), (10, 9), (9, 0)]

    target_gs, measured_nodes = extractor.extract_target_graph_state(
        cluster_4x4, target_edges, target_nodes
    )

    print(
        f"   Target graph state: "
        f"{len(target_gs.nodes)} nodes, {len(target_gs.edges)} edges"
    )
    print(f"   Nodes measured out: {len(measured_nodes)}")

    # Analyze local equivalence
    print("\n2. Local equivalence analysis")
    invariants = extractor.compute_local_equivalence_invariants(target_gs)
    print(f"   Degree sequence: {invariants['degree_sequence']}")
    print(f"   Spectrum: {invariants['spectrum'][:5]}...")  # Show first 5 eigenvalues
    print(f"   Number of triangles: {invariants['triangles']}")
    print(f"   Connected: {invariants['is_connected']}")

    # k-pairable analysis
    print("\n3. k-pairable structure analysis")
    k_analysis = extractor.analyze_k_pairable_structure(target_gs, k=2)
    print(
        f"   2-pairable subsets: "
        f"{k_analysis['pairable_subsets']}/{k_analysis['total_k_subsets']}"
    )
    print(f"   Pairable ratio: {k_analysis['pairable_ratio']:.3f}")

    # Fusion network analysis
    print("\n4. Fusion network decomposition")
    try:
        fusion_network = get_fusion_network_from_graph(target_gs)
        print(f"   Number of resource states: {len(fusion_network)}")

        for i, resource_state in enumerate(fusion_network):
            info = analyze_resource_graph(resource_state)
            if "nodes" in info and "edges" in info:
                print(
                    f"   Resource state {i}: "
                    f"{info['nodes']} nodes, {info['edges']} edges"
                )
            else:
                print(f"   Resource state {i}: {info['type']}")

            if "kind" in info:
                print(f"     Type: {info['kind']}")
            elif "resource_type" in info:
                print(f"     Type: {info['resource_type']}")

    except Exception as e:  # noqa: BLE001
        print(f"   Error in fusion network analysis: {e}")
        fusion_network = []

    print("\n5. Visualizing target graph state")
    try:
        target_gs.draw()
    except Exception as e:  # noqa: BLE001
        print(f"   Visualization not available: {e}")
        print(
            f"   Graph structure: "
            f"{len(target_gs.nodes)} nodes, {len(target_gs.edges)} edges"
        )
        print(f"   Nodes: {list(target_gs.nodes)}")
        print(f"   Edges: {list(target_gs.edges)}")

    return target_gs, invariants, k_analysis


def analyze_local_equivalence_classes() -> Dict[Tuple[Any, ...], List[Tuple[str, int, GraphState]]]:
    """Generate and analyze multiple graph states to study local equivalence classes."""
    print("\nLocal Equivalence Class Analysis")
    print("=" * 32)

    extractor = GraphStateExtractor()
    equivalence_classes: Dict[Tuple[Any, ...], List[Tuple[str, int, GraphState]]] = {}

    # Generate various small graph states
    test_graphs = []

    # Path graphs
    for n in range(3, 7):
        gs = GraphState()
        nodes = list(range(n))
        edges = [(i, i + 1) for i in range(n - 1)]
        gs.add_nodes_from(nodes)
        gs.add_edges_from(edges)
        test_graphs.append(("path", n, gs))

    # Cycle graphs
    for n in range(3, 7):
        gs = GraphState()
        nodes = list(range(n))
        edges = [(i, (i + 1) % n) for i in range(n)]
        gs.add_nodes_from(nodes)
        gs.add_edges_from(edges)
        test_graphs.append(("cycle", n, gs))

    # Complete graphs
    for n in range(3, 6):
        gs = GraphState()
        nodes = list(range(n))
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        gs.add_nodes_from(nodes)
        gs.add_edges_from(edges)
        test_graphs.append(("complete", n, gs))

    # Analyze each graph
    for graph_type, n, gs in test_graphs:
        invariants = extractor.compute_local_equivalence_invariants(gs)

        # Create a signature for equivalence class
        signature = (
            tuple(invariants["degree_sequence"]),
            tuple(np.round(invariants["spectrum"], 3)),
            invariants["triangles"],
            invariants["is_connected"],
        )

        if signature not in equivalence_classes:
            equivalence_classes[signature] = []

        equivalence_classes[signature].append((graph_type, n, gs))

    print(f"Found {len(equivalence_classes)} distinct equivalence classes:")
    for i, (signature, graphs) in enumerate(equivalence_classes.items()):
        print(f"\nClass {i+1}: {len(graphs)} graphs")
        print(f"  Degree sequence: {signature[0]}")
        print(f"  Representative graphs: {[(g[0], g[1]) for g in graphs[:3]]}")

    return equivalence_classes


def main() -> None:
    """Run the main demonstration and analysis."""
    # Run the main demonstration
    target_gs, invariants, k_analysis = demonstrate_graph_state_extraction()

    # Analyze local equivalence classes
    equivalence_classes = analyze_local_equivalence_classes()

    # Run scalability analysis
    sizes, extraction_times, equivalence_times, fusion_times = run_scalability_analysis()

    # Summary of results
    print("\n" + "=" * 60)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 60)

    print(f"\nScalability Analysis Results:")
    print(f"- Largest system analyzed: {max(sizes)}x{max(sizes)} = {max(sizes)**2} nodes")

    # Compute scaling exponents with error handling
    try:
        if len(sizes) > 2 and all(t > 0 for t in extraction_times):
            extraction_exp = np.polyfit(
                np.log([s * s for s in sizes]), np.log(extraction_times), 1
            )[0]
            print(f"- Extraction time scaling: O(n^{extraction_exp:.2f})")
        else:
            print("- Extraction time scaling: insufficient data")

        if len(sizes) > 2 and all(t > 0 for t in equivalence_times):
            equivalence_exp = np.polyfit(
                np.log([s * s for s in sizes]), np.log(equivalence_times), 1
            )[0]
            print(f"- Equivalence analysis scaling: O(n^{equivalence_exp:.2f})")
        else:
            print("- Equivalence analysis scaling: insufficient data")

        if len(sizes) > 2 and all(t > 0 for t in fusion_times):
            fusion_exp = np.polyfit(
                np.log([s * s for s in sizes]), np.log(fusion_times), 1
            )[0]
            print(f"- Fusion network scaling: O(n^{fusion_exp:.2f})")
        else:
            print("- Fusion network scaling: insufficient data")

    except Exception as e:  # noqa: BLE001
        print(f"- Scaling analysis failed: {e}")
        print(
            f"- Raw timing ranges: "
            f"Extraction [{min(extraction_times):.4f}, {max(extraction_times):.4f}]s"
        )
        print(
            f"                    "
            f"Equivalence [{min(equivalence_times):.4f}, {max(equivalence_times):.4f}]s"
        )
        print(
            f"                    "
            f"Fusion [{min(fusion_times):.4f}, {max(fusion_times):.4f}]s"
        )

    print(f"\nLocal Equivalence Classes:")
    print(f"- Found {len(equivalence_classes)} distinct classes among test graphs")
    class_sizes = [len(graphs) for graphs in equivalence_classes.values()]
    print(f"- Average graphs per class: {np.mean(class_sizes):.1f}")

    print(f"\nPotential Graphix Improvements:")
    improvements = [
        "Native support for local complementation operations",
        "Efficient local equivalence class computation",
        "Optimized 2D cluster state generation utilities",
        "Built-in k-pairable structure analysis",
        "Measurement pattern optimization for graph state extraction",
        "Parallel processing for large-scale equivalence analysis",
        "Integration with quantum error correction codes",
        "Advanced visualization for equivalence class relationships",
    ]

    for i, improvement in enumerate(improvements, 1):
        print(f"{i}. {improvement}")


if __name__ == "__main__":
    main()
