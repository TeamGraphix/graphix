"""
QAOA Interactive Visualization (Optimized)
==========================================

This example generates a QAOA pattern using the Graphix Circuit API
and launches the interactive visualizer in simulation-free mode
to demonstrate performance on complex patterns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np

# Add project root to path to ensure we use local graphix version
sys.path.insert(0, str(Path(__file__).parent.parent))

from graphix import Circuit
from graphix.visualization_interactive import InteractiveGraphVisualizer


def main() -> None:
    print("Generating QAOA pattern...")

    # 1. Define QAOA Circuit
    n_qubits = 4
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility

    # Random parameters for the circuit
    xi = rng.random(6)
    theta = rng.random(4)

    # Create a complete graph for the problem hamiltonian
    g = nx.complete_graph(n_qubits)
    circuit = Circuit(n_qubits)

    # Apply unitary evolution for the problem Hamiltonian
    for i, (u, v) in enumerate(g.edges):
        circuit.cnot(u, v)
        circuit.rz(v, xi[i])  # Rotation by random angle
        circuit.cnot(u, v)

    # Apply unitary evolution for the mixing Hamiltonian
    for v in g.nodes:
        circuit.rx(v, theta[v])

    # 2. Transpile to MBQC Pattern
    # This automatically generates the measurement pattern from the gate circuit
    pattern = circuit.transpile().pattern

    # Standardize the pattern to ensure it follows the standard MBQC form (N, E, M, C)
    pattern.standardize()
    pattern.shift_signals()

    print(f"Pattern generated with {len(pattern)} commands.")
    print("Launching interactive visualizer...")
    print("Optimization enabled: Simulation is DISABLED for performance.")
    print("You will see the graph structure and command flow without quantum state calculation.")

    # 3. Launch Visualization
    # enable_simulation=False prevents high RAM usage for this complex pattern
    viz = InteractiveGraphVisualizer(pattern, node_distance=(1.5, 1.5), enable_simulation=False)
    viz.visualize()


if __name__ == "__main__":
    main()
