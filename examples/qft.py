"""
Minimizing the pattern space
============================

Here, we demonstrate the effect of :meth:`graphix.pattern.Pattern.minimize_space()`.
This method reduces the maximum number of qubits that must be prepared at each step of MBQC operation,
by delaying the preparation and entanglement of qubits as much as the logical dependency structure allows.
This reduces the qubit count (or memory size for classical simulators such as graphix) requirement
for any quantum algorithms running on MBQC.

We will demonstrate this by simulating QFT on three qubits.
First, import relevant modules and define additional gates we'll use:
"""

import numpy as np
from graphix import Circuit
import networkx as nx
import matplotlib.pyplot as plt


def cp(circuit, theta, control, target):
    """Controlled rotation gate, decomposed
    """
    circuit.rz(control, theta / 2)
    circuit.rz(target, theta / 2)
    circuit.cnot(control, target)
    circuit.rz(target, -1 * theta / 2)
    circuit.cnot(control, target)


def swap(circuit, a, b):
    """swap gate, decomposed
    """
    circuit.cnot(a, b)
    circuit.cnot(b, a)
    circuit.cnot(a, b)


#%%
# Now let us define a circuit to apply QFT to three-qubit |011> state (input=6).


circuit = Circuit(3)
for i in range(3):
    circuit.h(i)

# prepare |011> state
circuit.x(1)
circuit.x(2)

# QFT
circuit.h(2)
cp(circuit, np.pi / 4, 0, 2)
cp(circuit, np.pi / 2, 1, 2)
circuit.h(1)
cp(circuit, np.pi / 2, 0, 1)
circuit.h(0)
swap(circuit, 0, 2)

# transpile and plot the graph
pattern = circuit.transpile()
nodes, edges = pattern.get_graph()
g = nx.Graph()
g.add_nodes_from(nodes)
g.add_edges_from(edges)
np.random.seed(100)
nx.draw(g)
plt.show()
print(len(nodes))

#%%
# This is a graph with 49 qubits, whose statevector is very hard to simulate in ordinary computers.
# As such, instead of preparing the graph state at the start of the compuation, we opt to prepare
# qubits as late as possible, so that (destructive) measurements will reduce the burden while we wait.
# For this, we first standardize and shift signals, to simplify the interdependence of measurements.
# After that, we can call :meth:`~graphix.pattern.Pattern.minimize_space()` to perform the optimization.

pattern.standardize()
pattern.shift_signals()
pattern.print_pattern(lim=20)
print(pattern.max_space())

#%%
# now compare with below:

pattern.minimize_space()
pattern.print_pattern(lim=20)
print(pattern.max_space())

#%%
# The maximum space has gone down to 4 which should be very easily simulated on laptops.
# Finally, we check the answer is correct, by comparing with statevector simulation.

out_state = pattern.simulate_pattern()
state = circuit.simulate_statevector()
print("overlap of states: ", np.abs(np.dot(state.psi.flatten().conjugate(), out_state.psi.flatten())))
