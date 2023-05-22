"""
Large-scale simulations with tensor network simulator
===================

In this example, we demonstrate the Tensor Network (TN) simulator to simulate MBQC
with up to ten-thousands of nodes.

You can also run this code on your browser with `mybinder.org <https://mybinder.org/>`_ - click the badge below.

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/TeamGraphix/graphix-examples/HEAD?labpath=qft_with_tn.ipynb

Firstly, let us import relevant modules:
"""

import numpy as np
from graphix import Circuit
import networkx as nx
import matplotlib.pyplot as plt

plt.rc("font", family="serif")


def cp(circuit, theta, control, target):
    circuit.rz(control, theta / 2)
    circuit.rz(target, theta / 2)
    circuit.cnot(control, target)
    circuit.rz(target, -1 * theta / 2)
    circuit.cnot(control, target)


def swap(circuit, a, b):
    circuit.cnot(a, b)
    circuit.cnot(b, a)
    circuit.cnot(a, b)


def qft_rotations(circuit, n):
    circuit.h(n)
    for qubit in range(n + 1, circuit.width):
        cp(circuit, np.pi / 2 ** (qubit - n), qubit, n)


def swap_registers(circuit, n):
    for qubit in range(n // 2):
        swap(circuit, qubit, n - qubit - 1)
    return circuit


def qft(circuit, n):
    for i in range(n):
        qft_rotations(circuit, i)
    swap_registers(circuit, n)


# %%
# We will simulate 50-qubit QFT, which requires graph states with more than 10000 nodes.

n = 50
print("{}-qubit QFT".format(n))
circuit = Circuit(n)

for i in range(n):
    circuit.h(i)
qft(circuit, n)

# standardize pattern
pattern = circuit.transpile()
pattern.standardize()
nodes, edges = pattern.get_graph()

print(f"Number of nodes: {len(nodes)}")
print(f"Number of edges: {len(edges)}")

# %%
# Using efficient graph state simulator `graphix.GraphSim`, we can classically preprocess Pauli measurements.

pattern.shift_signals()
pattern.perform_pauli_measurements()

nodes, edges = pattern.get_graph()
g = nx.Graph()
g.add_nodes_from(nodes)
g.add_edges_from(edges)
print(f"Number of nodes: {len(nodes)}")
print(f"Number of edges: {len(edges)}")
nx.draw(g, node_size=10)
plt.show()

# %%
# You can easily check that the below code run without too much load on your computer.
# Also notice that we have not used :meth:`graphix.pattern.Pattern.minimize_space()`,
# which we know reduced the burden on the simulator.
# To specify TN backend of the simulation, simply provide as a keyword argument.
# here we do a very basic check that the state is what it is expected to be:

tn = pattern.simulate_pattern(backend="tensornetwork")
value = tn.get_basis_amplitude(0)
print("amplitude of |00...0> is ", value)
print("1/2^n (true answer) is", 1 / 2**n)
