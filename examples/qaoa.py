"""
QAOA
====

Here we generate and optimize pattern for QAOA circuit.
You can run this code on your browser with `mybinder.org <https://mybinder.org/>`_ - click the badge below.

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/TeamGraphix/graphix-examples/HEAD?labpath=qaoa.ipynb


"""

# %%
import networkx as nx
import numpy as np

from graphix import Circuit

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

# %%
# transpile and get the graph state

pattern = circuit.transpile().pattern
pattern.standardize()
pattern.shift_signals()
pattern.draw_graph(flow_from_pattern=False)


# %%
# perform Pauli measurements and plot the new (minimal) graph to perform the same quantum computation

pattern.perform_pauli_measurements()
pattern.draw_graph(flow_from_pattern=False)

# %%
# finally, simulate the QAOA circuit

out_state = pattern.simulate_pattern()
state = circuit.simulate_statevector().statevec
print("overlap of states: ", np.abs(np.dot(state.psi.flatten().conjugate(), out_state.psi.flatten())))
# sphinx_gallery_thumbnail_number = 2

# %%
