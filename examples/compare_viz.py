"""
Visualization Comparison Demo
==============================
This script opens both the static and the new interactive visualizers
side-by-side (or one after another) using the exact same pattern 
so you can compare them visually as requested by the reviewers.
"""

from __future__ import annotations
import matplotlib.pyplot as plt

from graphix.command import E, M, N, X, Z
from graphix.measurements import Measurement
from graphix.pattern import Pattern
from graphix.visualization_interactive import InteractiveGraphVisualizer

def main() -> None:
    # Create the same simple pattern used in the interactive demo
    p = Pattern(input_nodes=[0, 1])
    p.add(N(node=2))
    p.add(E(nodes=(0, 2)))
    p.add(E(nodes=(1, 2)))
    p.add(M(node=0, measurement=Measurement.XY(0.5)))
    p.add(M(node=1, measurement=Measurement.XY(0.25)))
    p.add(X(node=2, domain={0, 1}))
    p.add(Z(node=2, domain={0}))

    print("Pattern created with", len(p), "commands.")
    print("Close the static plot window to open the interactive one.")
    
    # 1. Show the static visualizer plot
    p.draw_graph(flow_from_pattern=False, show_measurement_planes=True)
    plt.show()  # Blocks until closed
    
    # 2. Show the interactive visualizer plot
    viz = InteractiveGraphVisualizer(p)
    viz.visualize()

if __name__ == "__main__":
    main()
