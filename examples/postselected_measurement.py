"""
Example: Post-selected measurement in MBQC (prototype for non-unitary gate simulation)

This script demonstrates how to perform a measurement on a graph state and post-select on a specific outcome.
If post-selection is not natively supported by Graphix, this script will highlight the limitation.
"""

from graphix.pattern import Pattern
from graphix.simulator import PatternSimulator
import numpy as np
import networkx as nx
from graphix.command import N, E, M
from graphix.fundamentals import Plane
import warnings

warnings.filterwarnings("ignore", message="Couldn't find `optuna`", module="cotengra")
warnings.filterwarnings("ignore", message="Couldn't import `kahypar`", module="cotengra")
warnings.filterwarnings("ignore", message="Couldn't find `optuna`, `cmaes`, or `nevergrad`", module="cotengra")

# Prepare a simple 2-qubit graph state (Bell state)
inputs = [0, 1]
outputs = [0, 1]
pattern = Pattern(input_nodes=inputs, output_nodes=outputs)

pattern.add(E(nodes=(0, 1)))

# Add a measurement on qubit 0 in the X basis (angle=0, plane='XY')
pattern.add(M(node=0, plane=Plane.XY, angle=0))

# Post-selection: we want to keep only outcome 0 (|+> projection)
# Now supported natively via the 'postselect' argument!

# Simulate with post-selection on qubit 0 outcome 0
postselect = {0: 0}
state = pattern.simulate_pattern(postselect=postselect)

print("Post-selected state (outcome 0 on qubit 0):")
print(state)
if hasattr(state, 'psi'):
    print("Statevector:")
    print(state.psi)


pattern.draw_graph(save=True, filename="graph_state.png")


# Note: For robust and efficient post-selection, Graphix should support specifying desired measurement outcomes directly in the simulator.
# This would allow for deterministic, repeatable, and efficient non-unitary gate simulation as required by Azses, Ruhman, and Sela (2024).
