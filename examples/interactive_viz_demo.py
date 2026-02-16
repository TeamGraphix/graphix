from graphix.pattern import Pattern
from graphix.command import N, M, E, X, Z
from graphix.fundamentals import Plane
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from graphix.visualization_interactive import InteractiveGraphVisualizer


def main():
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
