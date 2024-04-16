"""
Statevector simulation of MBQC patterns
=======================================

Here we benchmark our statevector simulator for MBQC.

The methods and modules we use are the followings:
    1. :meth:`graphix.pattern.Pattern.simulate_pattern`
        Pattern simulator with statevector backend.
    2. :mod:`paddle_quantum.mbqc`
        Pattern simulation using :mod:`paddle_quantum.mbqc`.
"""

# %%
# Firstly, let us import relevant modules:

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from paddle import to_tensor
from paddle_quantum.mbqc.qobject import Circuit as PaddleCircuit
from paddle_quantum.mbqc.simulator import MBQC as PaddleMBQC
from paddle_quantum.mbqc.transpiler import transpile as PaddleTranspile

from graphix import Circuit

# %%
# Next, define a circuit to be transpiled into measurement pattern:


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
        np.random.shuffle(qubit_index)
        for j in range(len(qubit_index) // 2):
            circuit.cnot(qubit_index[2 * j], qubit_index[2 * j + 1])
        for j in range(len(qubit_index)):
            circuit.rz(qubit_index[j], 2 * np.pi * np.random.random())
    return circuit


# %%
# We define the test cases: shallow (depth=1) random circuits, only changing the number of qubits.

DEPTH = 1
test_cases = [i for i in range(2, 22)]
graphix_circuits = {}

pattern_time = []
circuit_time = []

# %%
# We then run simulations.
# First, we run the pattern simulation using `graphix`.
# For reference, we perform simple statevector simulation of the original gate network.
# Since transpilation into MBQC involves a significant increase in qubit number,
# the MBQC simulation is inherently slower as we will see.

for width in test_cases:
    circuit = simple_random_circuit(width, DEPTH)
    graphix_circuits[width] = circuit
    pattern = circuit.transpile()
    pattern.standardize()
    pattern.minimize_space()
    nodes, edges = pattern.get_graph()
    nqubit = len(nodes)
    start = perf_counter()
    pattern.simulate_pattern(max_qubit_num=30)
    end = perf_counter()
    print(f"width: {width}, nqubit: {nqubit}, depth: {DEPTH}, time: {end - start}")
    pattern_time.append(end - start)
    start = perf_counter()
    circuit.simulate_statevector()
    end = perf_counter()
    circuit_time.append(end - start)


# %%
# Here we benchmark `paddle_quantum`, using the same original gate network and use `paddle_quantum.mbqc` module
# to transpile into a measurement pattern.


def translate_graphix_rc_into_paddle_quantum_circuit(graphix_circuit: Circuit) -> PaddleCircuit:
    """Translate graphix circuit into paddle_quantum circuit.

    Parameters
    ----------
        graphix_circuit : Circuit
            graphix circuit

    Returns
    -------
        paddle_quantum_circuit : PaddleCircuit
            paddle_quantum circuit
    """
    paddle_quantum_circuit = PaddleCircuit(graphix_circuit.width)
    for instr in graphix_circuit.instruction:
        if instr[0] == "CNOT":
            paddle_quantum_circuit.cnot(which_qubits=instr[1])
        elif instr[0] == "Rz":
            paddle_quantum_circuit.rz(which_qubit=instr[1], theta=to_tensor(instr[2], dtype="float64"))
    return paddle_quantum_circuit


test_cases_for_paddle_quantum = [i for i in range(2, 22)]
paddle_quantum_time = []

for width in test_cases_for_paddle_quantum:
    graphix_circuit = graphix_circuits[width]
    paddle_quantum_circuit = translate_graphix_rc_into_paddle_quantum_circuit(graphix_circuit)
    pat = PaddleTranspile(paddle_quantum_circuit)
    mbqc = PaddleMBQC()
    mbqc.set_pattern(pat)
    start = perf_counter()
    mbqc.run_pattern()
    end = perf_counter()
    paddle_quantum_time.append(end - start)

    print(f"width: {width}, depth: {DEPTH}, time: {end - start}")

# %%
# Lastly, we compare the simulation times.

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(
    test_cases, circuit_time, label="direct statevector sim of original gate-based circuit (reference)", marker="x"
)
ax.scatter(test_cases, pattern_time, label="graphix pattern simulator")
ax.scatter(test_cases_for_paddle_quantum, paddle_quantum_time, label="paddle_quantum pattern simulator")
ax.set(
    xlabel="Width of the original circuit",
    ylabel="time (s)",
    yscale="log",
    title="Time to simulate random circuits",
)
fig.legend(bbox_to_anchor=(0.85, 0.9))
fig.show()

# %%
# MBQC simulation is a lot slower than the simulation of original gate network, since the number of qubit involved
# is significantly larger.

import importlib.metadata  # noqa: E402

# print package versions.
[
    print("{} - {}".format(pkg, importlib.metadata.version(pkg)))
    for pkg in ["numpy", "graphix", "paddlepaddle", "paddle-quantum"]
]
