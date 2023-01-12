"""
Using MPS simulator
===================

In this example, we demonstrate the matrix product state (MPS) simulator to simulate MBQC
with up to thousands of nodes at a time, without the need for approximation which is often present for circuit-MPS simulators.

You can run this code on your browser with `mybinder.org <https://mybinder.org/>`_ - click the badge below.

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/TeamGraphix/graphix-examples/HEAD?labpath=qft_with_mps.ipynb


We will simulate n-qubit QFT circuit.
Firstly, let us import relevant modules:
"""

import numpy as np
from graphix import Circuit
import networkx as nx
import matplotlib.pyplot as plt


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
    if n == 0:
        return circuit
    n -= 1
    circuit.h(n)
    for qubit in range(n):
        cp(circuit, np.pi / 2 ** (n - qubit), qubit, n)


def swap_registers(circuit, n):
    for qubit in range(n // 2):
        swap(circuit, qubit, n - qubit - 1)
    return circuit


def qft(circuit, n):
    for i in range(n):
        m = n - i
        qft_rotations(circuit, m)
    swap_registers(circuit, n)


#%%
# We will simulate 7-qubit QFT, which requires nearly 250 nodes to be simulated.

n = 7
print("{}-qubit QFT".format(n))
circuit = Circuit(n)

for i in range(n):
    circuit.h(i)
qft(circuit, n)

# standardize pattern
pattern = circuit.transpile()
pattern.standardize()
nodes, edges = pattern.get_graph()
g = nx.Graph()
g.add_nodes_from(nodes)
g.add_edges_from(edges)
np.random.seed(100)
nx.draw(g)
plt.show()
print(len(nodes))

#%%
# You can easily check that the below code run without much load on the computer.
# Also notice that we have not used :meth:`graphix.pattern.Pattern.minimize_space()`,
# which we know reduced the burden on the simulator.
# To specify MPS backend of the simulation, simply provide as a keyword argument.
# here we do a very basic check that the state is what is is expected to be:

mps = pattern.simulate_pattern(backend="mps")
value = mps.get_amplitude(0)
print("amplitude of |00000> is ", value)
print("1/2^n (true answer) is", 1 / 2 ** n)
