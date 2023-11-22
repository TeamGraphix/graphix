"""
Statevector simulation of MBQC patterns
=======================================

Here we benchmark our statevector simulator for MBQC.

The methods and modules we use are the followings:
    1. :meth:`~graphix.pattern.Pattern.simulate_pattern`
        Pattern simulator with statevector backend.
    2. :mod:`paddle_quantum.mbqc`
        Pattern simulation using :mod:`paddle_quantum.mbqc`.
    3. :mod:`mentpy`
        Pattern simulation using :mod:`mentpy`.
"""

# %%
# Firstly, let us import relevant modules:

from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
from graphix import Circuit
from graphix.sim.statevec import StatevectorBackend
from paddle import to_tensor
from paddle_quantum.mbqc.qobject import Circuit as PaddleCircuit
from paddle_quantum.mbqc.transpiler import transpile as PaddleTranspile
from paddle_quantum.mbqc.simulator import MBQC as PaddleMBQC

# %%
# Next, we define a random circuit generator:


def simple_random_circuit(nqubit, depth):
    r"""Generate a random circuit.

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
# First, we run the pattern simulations and circuit simulation.

for nqubit in test_cases:
    circuit = simple_random_circuit(nqubit, DEPTH)
    graphix_circuits[nqubit] = circuit
    pattern = circuit.transpile()
    pattern.standardize()
    pattern.minimize_space()
    max_qubit_num = 20 if nqubit < 20 else 50
    backend = StatevectorBackend(pattern, max_qubit_num=max_qubit_num)
    print(f"max space for nqubit={nqubit} circuit is ", max_qubit_num)
    start = perf_counter()
    backend.pattern.simulate_pattern(max_qubit_num=max_qubit_num)
    end = perf_counter()
    print(f"nqubit: {nqubit}, depth: {DEPTH}, time: {end - start}")
    pattern_time.append(end - start)
    start = perf_counter()
    circuit.simulate_statevector()
    end = perf_counter()
    circuit_time.append(end - start)


# %%
# Here we take benchmarking for MBQC simulation using `paddle_quantum`.


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
            paddle_quantum_circuit.rz(
                which_qubit=instr[1], theta=to_tensor(instr[2], dtype="float64"))
    return paddle_quantum_circuit


test_cases_for_paddle_quantum = [i for i in range(2, 22)]
paddle_quantum_time = []

for width in test_cases_for_paddle_quantum:
    graphix_circuit = graphix_circuits[width]
    paddle_quantum_circuit = translate_graphix_rc_into_paddle_quantum_circuit(
        graphix_circuit)
    pat = PaddleTranspile(paddle_quantum_circuit)
    mbqc = PaddleMBQC()
    mbqc.set_pattern(pat)
    start = perf_counter()
    mbqc.run_pattern()
    end = perf_counter()
    paddle_quantum_time.append(end - start)

    print(f"nqubit: {width}, depth: {DEPTH}, time: {end - start}")

# %%
# Lastly, we compare the simulation times.

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(
    test_cases, circuit_time,
    label="direct statevector sim of original gate-based circuit (for reference)", marker="x"
)
ax.scatter(test_cases, pattern_time, label="graphix pattern simulator")
ax.scatter(test_cases_for_paddle_quantum, paddle_quantum_time,
           label="paddle_quantum pattern simulator")
ax.set(
    xlabel="nqubit",
    ylabel="time (s)",
    yscale="log",
    title="Time to simulate random circuits",
)
fig.legend(bbox_to_anchor=(0.85, 0.9))
fig.show()

# %%
