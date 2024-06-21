"""
Using Tensor Network simulator
==============================

In this example, we simulate a circuit to create Greenberger-Horne-Zeilinger(GHZ) state with a tensor network simulator.

You can run this code on your browser with `mybinder.org <https://mybinder.org/>`_ - click the badge below.

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/TeamGraphix/graphix-examples/HEAD?labpath=ghz_with_tn.ipynb

We will simulate the generation of 100-qubit GHZ state.
Firstly, let us import relevant modules:
"""

# %%

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from graphix import Circuit

n = 100
print(f"{n}-qubit GHZ state generation")
circuit = Circuit(n)

# initialize to ``|0>`` state.
for i in range(n):
    circuit.h(i)

# GHZ generation
circuit.h(0)
for i in range(1, n):
    circuit.cnot(i - 1, i)

# %%
# Transpile into pattern

pattern = circuit.transpile().pattern
pattern.standardize()

nodes, edges = pattern.get_graph()
g = nx.Graph()
g.add_nodes_from(nodes)
g.add_edges_from(edges)
print(f"Number of nodes: {len(nodes)}")
print(f"Number of edges: {len(edges)}")
np.random.seed(100)
pos = nx.spring_layout(g)
nx.draw(g, pos=pos, node_size=15)
plt.show()

# %%
# Calculate the amplitudes of ``|00...0>`` and ``|11...1>`` states.

tn = pattern.simulate_pattern(backend="tensornetwork")
print(f"The amplitude of |00...0>: {tn.get_basis_amplitude(0)}")
print(f"The amplitude of |11...1>: {tn.get_basis_amplitude(2**n-1)}")

# %%
