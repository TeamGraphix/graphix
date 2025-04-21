"""
Large-scale simulations with tensor network simulator
=====================================================

In this example, we demonstrate simulation of MBQC involving 10k+ nodes.

Firstly, let us import relevant modules and define the circuit:
"""

# %%
from __future__ import annotations

import numpy as np

from graphix import Circuit


def cp(circuit, theta, control, target):
    circuit.rz(control, theta / 2)
    circuit.rz(target, theta / 2)
    circuit.cnot(control, target)
    circuit.rz(target, -1 * theta / 2)
    circuit.cnot(control, target)


def qft_rotations(circuit, n):
    circuit.h(n)
    for qubit in range(n + 1, circuit.width):
        cp(circuit, np.pi / 2 ** (qubit - n), qubit, n)


def swap_registers(circuit, n):
    for qubit in range(n // 2):
        circuit.swap(qubit, n - qubit - 1)
    return circuit


def qft(circuit, n):
    for i in range(n):
        qft_rotations(circuit, i)
    swap_registers(circuit, n)


# %%
# We will simulate 55-qubit QFT, which requires graph states with more than 10000 nodes.

n = 55
print(f"{n}-qubit QFT")
circuit = Circuit(n)

for i in range(n):
    circuit.h(i)
qft(circuit, n)

# standardize pattern
pattern = circuit.transpile().pattern
pattern.standardize()
pattern.shift_signals()
nodes, edges = pattern.get_graph()
print(f"Number of nodes: {len(nodes)}")
print(f"Number of edges: {len(edges)}")

# %%
# Using efficient graph state simulator `graphix.graphsim`, we can classically preprocess Pauli measurements.
# We are currently improving the speed of this process by using rust-based graph manipulation backend.
pattern.perform_pauli_measurements()


# %%
# To specify TN backend of the simulation, simply provide as a keyword argument.
# here we do a very basic check that one of the statevector amplitudes is what it is expected to be:

import time

t1 = time.time()
tn = pattern.simulate_pattern(backend="tensornetwork")
value = tn.get_basis_amplitude(0)
t2 = time.time()
print("amplitude of |00...0> is ", value)
print("1/2^n (true answer) is", 1 / 2**n)
print("approximate execution time in seconds: ", t2 - t1)

# %%
