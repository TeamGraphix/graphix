"""
Optimized pattern fragments
===========================

Graphix offers several pattern fragments for quantum gates, pre-optimized by reducing the Pauli measurements,
with fewer resource requirement than the standard pattern.

"""

# %%
# First, for Toffoli gate, here is the pattern based on the decomposition of CCX gate with CNOT and single-qubit rotations,
# turned into a measurement pattern:
from __future__ import annotations

import numpy as np

from graphix import Circuit

circuit = Circuit(3)
circuit.ccx(0, 1, 2)
pattern = circuit.transpile().pattern
pattern.draw_graph(flow_from_pattern=False)

# %%
# Using :code:`opt=True` option for :code:`transpile` method, we switch to patterns with non-XY plane measurements allowed,
# which has gflow (not flow). For CCX gate, the number of ancilla qubits required is nearly halved:
pattern = circuit.transpile(opt=True).pattern
pattern.draw_graph(node_distance=(1.2, 0.8))
# sphinx_gallery_thumbnail_number = 2

# %%
# Now let us add z-rotation gates, requiring two ancillas in the original pattern fragment,
# which becomes one for patterns with :code:`opt=True`.
circuit = Circuit(3)
circuit.ccx(0, 1, 2)
for i in range(3):
    circuit.rz(i, np.pi / 4)
pattern = circuit.transpile(opt=True).pattern
pattern.draw_graph(flow_from_pattern=True, node_distance=(1, 0.5))

# %%
# Swap gate is just a swap of node indices during compilation, requiring no ancillas.
circuit = Circuit(3)
circuit.ccx(0, 1, 2)
circuit.swap(1, 2)
circuit.swap(2, 0)
for i in range(3):
    circuit.rz(i, np.pi / 4)
pattern = circuit.transpile(opt=True).pattern
pattern.draw_graph(flow_from_pattern=False, node_distance=(1, 0.4))


# %%
# using :code:`opt=True` and with either CCX, Rzz or Rz gates, the graph will have gflow.
circuit = Circuit(4)
circuit.cnot(1, 2)
circuit.cnot(0, 3)
circuit.ccx(2, 1, 0)
circuit.rx(0, np.pi / 3)
circuit.cnot(0, 3)
circuit.rzz(0, 3, np.pi / 3)
circuit.rx(2, np.pi / 3)
circuit.ccx(3, 1, 2)
circuit.rx(0, np.pi / 3)
circuit.rx(3, np.pi / 3)
pattern = circuit.transpile(opt=True).pattern
pattern.draw_graph(flow_from_pattern=False, node_distance=(1, 0.4))


# %%
# reducing the size further
pattern.perform_pauli_measurements()
pattern.draw_graph(flow_from_pattern=False, node_distance=(1, 0.6))

# %%
# For linear optical QPUs with single photons, such a resource state can be generated by fusing the follwoing microcluster states:
from graphix.extraction import get_fusion_network_from_graph

nodes, edges = pattern.get_graph()
from graphix import GraphState

gs = GraphState(nodes=nodes, edges=edges)
get_fusion_network_from_graph(gs, max_ghz=4, max_lin=4)

# %%
