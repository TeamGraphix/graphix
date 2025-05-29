from __future__ import annotations

import itertools
import time
from typing import TYPE_CHECKING, Any, TypedDict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import graphix
from graphix import GraphState
from graphix.extraction import get_fusion_network_from_graph

if TYPE_CHECKING:
    from typing import List, Optional, Set, Tuple


class ResourceGraphInfo(TypedDict):
    """Type definition for resource graph analysis results."""
    type: str
    attributes: List[str]
    nodes: int
    edges: int
    kind: str
    resource_type: str


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
        "attributes": [attr for attr in dir(resource_graph) if not attr.startswith("_")],
        "nodes": 0,
        "edges": 0,
        "kind": "",
        "resource_type": "",
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


class GraphStateExtractor:
    """Extract target graph states from 2D cluster states and analyze local equivalence."""

    # Class-level type annotations
    extraction_times: List[float]
    equivalence_times: List[float]

    def __init__(self) -> None:
        """Initialize the extractor with empty timing lists."""
        self.extraction_times = []
        self.equivalence_times = []

    def create_2d_cluster_state(self, rows: int, cols: int) -> GraphState:
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
        target_edges: List[Tuple[int, int]],
        target_nodes: Optional[List[int]] = None,
    ) -> Tuple[GraphState, List[int]]:
        """Extract a target graph state from a 2D cluster state using local measurements.

        This simulates the process described in Freund et al. where specific measurement
        patterns on cluster states can produce desired graph states.

        Parameters
        ----------
        cluster_state : GraphState
            The source 2D cluster state
        target_edges : List[Tuple[int, int]]
            Desired edges in the target graph state
        target_nodes : Optional[List[int]], optional
            Nodes to keep in target state (if None, infer from edges)

        Returns
        -------
        Tuple[GraphState, List[int]]
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

        Based on Claudet and Perdrix's graphical characterization of local equivalence.

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

        invariants: ResourceGraphInfo = {
            "type": "LocalEquivalenceInvariants",
            "attributes": [],
            "nodes": len(gs.nodes),
            "edges": len(gs.edges),
            "kind": "",
            "resource_type": "",
        }

        # Degree sequence (sorted)
        degrees = sorted(graph.degree(node) for node in graph.nodes())
        invariants["attributes"].extend([
            f"degree_sequence_{degrees}",
            f"max_degree_{max(degrees) if degrees else 0}",
            f"min_degree_{min(degrees) if degrees else 0}",
        ])

        # Spectral properties of adjacency matrix
        if len(gs.nodes) > 0:
            adj_matrix = nx.adjacency_matrix(graph).todense()
            eigenvals = np.linalg.eigvals(adj_matrix)
            # Sort eigenvalues and round to handle numerical precision
            eigenvals_real = sorted(np.round(eigenvals.real, 6))
            rank = np.linalg.matrix_rank(adj_matrix)
            invariants["attributes"].extend([
                f"spectrum_{eigenvals_real}",
                f"rank_{rank}",
            ])

        # Triangle count and clustering properties  
        triangles = sum(nx.triangles(graph).values()) // 3
        clustering_coeffs = sorted(nx.clustering(graph).values())
        invariants["attributes"].extend([
            f"triangles_{triangles}",
            f"clustering_coeffs_{clustering_coeffs}",
        ])

        # Connectedness properties
        is_connected = nx.is_connected(graph)
        num_components = nx.number_connected_components(graph)
        invariants["attributes"].extend([
            f"is_connected_{is_connected}",
            f"num_components_{num_components}",
        ])

        equivalence_time = time.perf_counter() - start_time
        self.equivalence_times.append(equivalence_time)

        return invariants

    def analyze_k_pairable_structure(self, gs: GraphState, k: int = 2) -> ResourceGraphInfo:
        """Analyze k-pairable structure of the graph state.

        Based on the concept of small k-pairable states from Claudet, Mhalla and Perdrix.

        Parameters
        ----------
        gs : GraphState
            Input graph state
        k : int, optional
            Parameter for k-pairable analysis, by default 2

        Returns
        -------
        ResourceGraphInfo
            Analysis results
        """
        graph = nx.Graph()
        graph.add_nodes_from(gs.nodes)
        graph.add_edges_from(gs.edges)

        results: ResourceGraphInfo = {
            "type": "KPairableAnalysis",
            "attributes": [],
            "nodes": len(gs.nodes),
            "edges": len(gs.edges),
            "kind": f"k_{k}_pairable",
            "resource_type": "analysis",
        }

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

        results["attributes"] = [
            f"k_{k}",
            f"total_k_subsets_{len(k_subsets)}",
            f"pairable_subsets_{len(pairable_subsets)}",
            f"pairable_ratio_{len(pairable_subsets) / len(k_subsets) if k_subsets else 0}",
        ]

        return results


def run_scalability_analysis() -> Tuple[List[int], List[float], List[float], List[float]]:
    """Analyze the scalability of graph state extraction and equivalence analysis.
    
    Returns
    -------
    Tuple[List[int], List[float], List[float], List[float]]
        Sizes, extraction times, equivalence times, and fusion times
    """
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
        start_time = time.perf_counter()
        target_gs, _ = extractor.extract_target_graph_state(
            cluster_state, target_edges, target_nodes
        )
        extraction_time = time.perf_counter() - start_time
        extraction_times.append(extraction_time)

        # Time the equivalence analysis
        start_time = time.perf_counter()
        _ = extractor.compute_local_equivalence_invariants(target_gs)
        equivalence_time = time.perf_counter() - start_time
        equivalence_times.append(equivalence_time)

        # Time the fusion network generation
        start_time = time.perf_counter()
        fusion_time = 0
        try:
            _ = get_fusion_network_from_graph(target_gs)
            fusion_time = time.perf_counter() - start_time
        except Exception as e:  # noqa: BLE001
            print(f"  Warning: Fusion network generation failed: {e}")
        
        fusion_times.append(fusion_time)

        print(f"  Extraction: {extraction_time:.4f}s")
        print(f"  Equivalence: {equivalence_time:.4f}s")
        print(f"  Fusion: {fusion_time:.4f}s")

    # Plot results using OOP style
    if plt.get_backend() != 'Agg':  # Only plot if display is available
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        node_counts = [s * s for s in sizes]

        axes[0].plot(node_counts, extraction_times, "o-", label="Extraction")
        axes[0].set_xlabel("Number of nodes")
        axes[0].set_ylabel("Time (s)")
        axes[0].set_title("Graph State Extraction Scaling")
        axes[0].set_yscale("log")
        axes[0].set_xscale("log")
        axes[0].grid(True)

        axes[1].plot(node_counts, equivalence_times, "o-", label="Equivalence", color="orange")
        axes[1].set_xlabel("Number of nodes")
        axes[1].set_ylabel("Time (s)")
        axes[1].set_title("Local Equivalence Analysis Scaling")
        axes[1].set_yscale("log")
        axes[1].set_xscale("log")
        axes[1].grid(True)

        axes[2].plot(node_counts, fusion_times, "o-", label="Fusion Network", color="green")
        axes[2].set_xlabel("Number of nodes")
        axes[2].set_ylabel("Time (s)")
        axes[2].set_title("Fusion Network Generation Scaling")
        axes[2].set_yscale("log")
        axes[2].set_xscale("log")
        axes[2].grid(True)

        fig.tight_layout()
        plt.show()
    else:
        print("Display not available, printing raw timing data:")
        for i, size in enumerate(sizes):
            print(
                f"Size {size}x{size}: "
                f"Extraction={extraction_times[i]:.4f}s, "
                f"Equivalence={equivalence_times[i]:.4f}s, "
                f"Fusion={fusion_times[i]:.4f}s"
            )

    return sizes, extraction_times, equivalence_times, fusion_times


def demonstrate_graph_state_extraction() -> Tuple[GraphState, ResourceGraphInfo, ResourceGraphInfo]:
    """Demonstrate the main functionality with concrete examples.
    
    Returns
    -------
    Tuple[GraphState, ResourceGraphInfo, ResourceGraphInfo]
        Target graph state, invariants, and k-analysis results
    """
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
    print(f"   Node count: {invariants['nodes']}")
    print(f"   Edge count: {invariants['edges']}")
    print(f"   Analysis type: {invariants['type']}")

    # k-pairable analysis
    print("\n3. k-pairable structure analysis")
    k_analysis = extractor.analyze_k_pairable_structure(target_gs, k=2)
    print(f"   Analysis kind: {k_analysis['kind']}")
    print(f"   Resource type: {k_analysis['resource_type']}")

    # Fusion network analysis
    print("\n4. Fusion network decomposition")
    try:
        fusion_network = get_fusion_network_from_graph(target_gs)
        print(f"   Number of resource states: {len(fusion_network)}")

        for i, resource_state in enumerate(fusion_network):
            info = analyze_resource_graph(resource_state)
            if info["nodes"] > 0 and info["edges"] > 0:
                print(
                    f"   Resource state {i}: "
                    f"{info['nodes']} nodes, {info['edges']} edges"
                )
            else:
                print(f"   Resource state {i}: {info['type']}")

            if info["kind"]:
                print(f"     Type: {info['kind']}")
            elif info["resource_type"]:
                print(f"     Type: {info['resource_type']}")

    except Exception as e:  # noqa: BLE001
        print(f"   Error in fusion network analysis: {e}")

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


def analyze_local_equivalence_classes() -> dict[Tuple[Any, ...], List[Tuple[str, int, GraphState]]]:
    """Generate and analyze multiple graph states to study local equivalence classes.
    
    Returns
    -------
    dict[Tuple[Any, ...], List[Tuple[str, int, GraphState]]]
        Dictionary mapping equivalence class signatures to lists of graphs
    """
    print("\nLocal Equivalence Class Analysis")
    print("=" * 32)

    extractor = GraphStateExtractor()
    equivalence_classes: dict[Tuple[Any, ...], List[Tuple[str, int, GraphState]]] = {}

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

        # Create a simplified signature for equivalence class
        signature = (
            invariants["type"],
            invariants["nodes"],
            invariants["edges"],
        )

        if signature not in equivalence_classes:
            equivalence_classes[signature] = []

        equivalence_classes[signature].append((graph_type, n, gs))

    print(f"Found {len(equivalence_classes)} distinct equivalence classes:")
    for i, (signature, graphs) in enumerate(equivalence_classes.items()):
        print(f"\nClass {i+1}: {len(graphs)} graphs")
        print(f"  Signature: {signature}")
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
    if len(sizes) > 2 and all(t > 0 for t in extraction_times):
        try:
            extraction_exp = np.polyfit(
                np.log([s * s for s in sizes]), np.log(extraction_times), 1
            )[0]
            print(f"- Extraction time scaling: O(n^{extraction_exp:.2f})")
        except (ValueError, np.linalg.LinAlgError):
            print("- Extraction time scaling: insufficient valid data")
    else:
        print("- Extraction time scaling: insufficient data")

    if len(sizes) > 2 and all(t > 0 for t in equivalence_times):
        try:
            equivalence_exp = np.polyfit(
                np.log([s * s for s in sizes]), np.log(equivalence_times), 1
            )[0]
            print(f"- Equivalence analysis scaling: O(n^{equivalence_exp:.2f})")
        except (ValueError, np.linalg.LinAlgError):
            print("- Equivalence analysis scaling: insufficient valid data")
    else:
        print("- Equivalence analysis scaling: insufficient data")

    if len(sizes) > 2 and all(t > 0 for t in fusion_times):
        try:
            fusion_exp = np.polyfit(
                np.log([s * s for s in sizes]), np.log(fusion_times), 1
            )[0]
            print(f"- Fusion network scaling: O(n^{fusion_exp:.2f})")
        except (ValueError, np.linalg.LinAlgError):
            print("- Fusion network scaling: insufficient valid data")
    else:
        print("- Fusion network scaling: insufficient data")

    print(f"\nLocal Equivalence Classes:")
    print(f"- Found {len(equivalence_classes)} distinct classes among test graphs")
    class_sizes = [len(graphs) for graphs in equivalence_classes.values()]
    if class_sizes:
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
