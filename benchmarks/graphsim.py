"""
Graph state simulator backends
=======================================

Here we benchmark our graph state simulator for MBQC with different backends.

Currently, we have two backends: `networkx <https://networkx.org/documentation/stable/index.html>`_
and `rustworkx <https://qiskit.org/ecosystem/rustworkx/index.html>`_.
Both Python packages are used to manipulate graphs.
While networkx is a pure Python package, rustworkx is a Rust package with Python bindings, which is faster than networkx.
"""

# %%
# Firstly, let us import relevant modules:

from copy import copy
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from graphix import Circuit

# %%
# Next, define a circuit to be transpiled into measurement pattern:

rng = np.random.default_rng(42)


def simple_random_circuit(nqubit, depth):
    r"""Generate a test circuit for benchmarking.

    This function generates a circuit with nqubit qubits and depth layers,
    having layers of CNOT and Rz gates with random placements.

    Parameters
    ----------
    nqubit : int
        number of qubits
    depth : int
        number of layers

    Returns
    -------
    circuit : graphix.transpiler.Circuit object
        generated circuit
    """
    qubit_index = [i for i in range(nqubit)]
    circuit = Circuit(nqubit)
    for _ in range(depth):
        rng.shuffle(qubit_index)
        for j in range(len(qubit_index) // 2):
            circuit.cnot(qubit_index[2 * j], qubit_index[2 * j + 1])
        for j in range(len(qubit_index)):
            circuit.rz(qubit_index[j], 2 * np.pi * rng.random())
    return circuit


# %%
# We define the test cases

test_cases = [i for i in range(1, 50, 5)]
graphix_patterns = {}

for i in test_cases:
    circuit = simple_random_circuit(i, i)
    pattern = circuit.transpile()
    pattern.standardize()
    graphix_patterns[i] = pattern


# %%
# We then run simulations.
# First, we run the pattern optimization using `networkx`.
networkx_time = []
networkx_node = []

for i, pattern in graphix_patterns.items():
    pattern_copy = copy(pattern)
    start = perf_counter()
    pattern_copy.perform_pauli_measurements()
    end = perf_counter()
    nodes, edges = pattern_copy.get_graph()
    num_nodes = len(nodes)
    networkx_node.append(num_nodes)
    print(f"width: {i}, number of nodes: {num_nodes}, depth: {i}, time: {end - start}")
    networkx_time.append(end - start)


# %%
# Next, we run the pattern optimization using `rustworkx`.
rustworkx_time = []
rustworkx_node = []

for i, pattern in graphix_patterns.items():
    pattern_copy = copy(pattern)
    start = perf_counter()
    pattern_copy.perform_pauli_measurements()
    end = perf_counter()
    nodes, edges = pattern_copy.get_graph()
    num_nodes = len(nodes)
    rustworkx_node.append(num_nodes)
    print(f"width: {i}, number of nodes: {num_nodes}, depth: {i}, time: {end - start}")
    rustworkx_time.append(end - start)

# %%
# Lastly, we compare the simulation times.
assert networkx_node == rustworkx_node

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(networkx_node, networkx_time, label="networkx", color="blue")
ax.scatter(rustworkx_node, rustworkx_time, label="rustworkx", color="red")
ax.set(
    xlabel="Number of nodes in the graph state",
    xscale="log",
    ylabel="Time (s)",
    yscale="log",
    title="Time to perform Pauli measurements on the graph state",
)
ax.legend()
fig.show()

# %%
# MBQC simulation is a lot slower than the simulation of original gate network, since the number of qubit involved
# is significantly larger.

import importlib.metadata  # noqa: E402

# print package versions.
[print("{} - {}".format(pkg, importlib.metadata.version(pkg))) for pkg in ["graphix", "networkx", "rustworkx"]]
