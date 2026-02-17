"""
Interactive Visualization Demo
============================

This example demonstrates the interactive graph visualizer using a simple
manually constructed pattern. It shows how to step through the visualization
and observe state changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

from graphix.command import E, M, N, X, Z
from graphix.fundamentals import Plane
from graphix.pattern import Pattern

sys.path.insert(0, str(Path(__file__).parent.parent))
from graphix.visualization_interactive import InteractiveGraphVisualizer


def main() -> None:
    # optimized pattern for QFT
    # Create a simple pattern manually for demonstration
    p = Pattern(input_nodes=[0, 1])
    p.add(N(node=2))
    p.add(E(nodes=(0, 2)))
    p.add(E(nodes=(1, 2)))
    p.add(M(node=0, plane=Plane.XY, angle=0.5))
    p.add(M(node=1, plane=Plane.XY, angle=0.25))
    p.add(X(node=2, domain={0, 1}))
    p.add(Z(node=2, domain={0}))

    # Or standardization to make it interesting
    # p.standardize()

    print("Pattern created with", len(p), "commands.")
    print("Launching interactive visualization with real-time simulation...")

    viz = InteractiveGraphVisualizer(p)
    viz.visualize()


if __name__ == "__main__":
    main()
