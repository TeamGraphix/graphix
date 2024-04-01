"""
Efficient contraction order statevector simulator
=================================================

In this minimal example, we show how to use the ``eco-statevec`` backend for the simulation of our MBQC pattern.

Here we define a QAOA circuit and simulate it using the ``eco-statevec`` backend.
"""

# %%
import networkx as nx
import numpy as np

from graphix import Circuit
from graphix.simulator import PatternSimulator

# QAOA circuit
n = 4
xi = np.random.rand(6)
theta = np.random.rand(4)
g = nx.complete_graph(n)
circuit = Circuit(n)
for i, (u, v) in enumerate(g.edges):
    circuit.cnot(u, v)
    circuit.rz(v, xi[i])
    circuit.cnot(u, v)
for v in g.nodes:
    circuit.rx(v, theta[v])

pattern = circuit.transpile()
pattern.standardize()

# %%
# Simulate the pattern for QAOA using eco-statevec backend

sim = PatternSimulator(pattern, backend="eco-statevec")
state = sim.run()

# %%
# Get the statevector. This method uses quimb to contract the tensor network.
print(state.to_statevector())
