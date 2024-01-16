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
# Next, define a function to generate random Clifford circuits.


def genpair(n_qubits, count, rng):
    pairs = []
    for i in range(count):
        choice = [j for j in range(n_qubits)]
        x = rng.choice(choice)
        choice.pop(x)
        y = rng.choice(choice)
        pairs.append((x, y))
    return pairs


def random_clifford_circuit(nqubits, depth, seed=42):
    rng = np.random.default_rng(seed)
    circuit = Circuit(nqubits)
    gate_choice = list(range(5))
    for _ in range(depth):
        for j, k in genpair(nqubits, 2, rng):
            circuit.cnot(j, k)
        for j in range(nqubits):
            k = rng.choice(gate_choice)
            if k == 0:  # H
                circuit.h(j)
            elif k == 1:  # S
                circuit.s(j)
            elif k == 2:  # X
                circuit.x(j)
            elif k == 3:  # Z
                circuit.z(j)
            elif k == 4:  # Y
                circuit.y(j)
            else:
                pass
    return circuit


# %%
# We generate a set of random Clifford circuits with different widths.

DEPTH = 3
test_cases = [i for i in range(2, 300, 10)]
graphix_patterns = {}

for i in test_cases:
    circuit = random_clifford_circuit(i, DEPTH)
    pattern = circuit.transpile()
    pattern.standardize()
    pattern.shift_signals()
    nodes, edges = pattern.get_graph()
    graphix_patterns[i] = (circuit, pattern, len(nodes))


# %%
# We then run simulations.
# First, we run the pattern optimization using networkx.
networkx_time = []
networkx_node = []

for width, (circuit, pattern, num_nodes) in graphix_patterns.items():
    pattern_copy = copy(pattern)
    start = perf_counter()
    pattern_copy.perform_pauli_measurements()
    end = perf_counter()
    networkx_node.append(num_nodes)
    print(f"width: {width}, number of nodes: {num_nodes}, depth: {DEPTH}, time: {end - start}")
    networkx_time.append(end - start)


# %%
# Next, we run the pattern optimization using rustworkx.
rustworkx_time = []
rustworkx_node = []

for width, (circuit, pattern, num_nodes) in graphix_patterns.items():
    pattern_copy = copy(pattern)
    start = perf_counter()
    pattern_copy.perform_pauli_measurements(use_rustworkx=True)
    end = perf_counter()
    rustworkx_node.append(num_nodes)
    print(f"width: {width}, number of nodes: {num_nodes}, depth: {DEPTH}, time: {end - start}")
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
# Performing pattern optimization using rustworkx is slightly faster than networkx.

import importlib.metadata  # noqa: E402

# print package versions.
for pkg in ["graphix", "networkx", "rustworkx"]:
    print("{} - {}".format(pkg, importlib.metadata.version(pkg)))
