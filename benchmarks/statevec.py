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
import mentpy as mp

# %%
# Next, we define the simulation runner:


def run(backend):
    """Perform MBQC simulation"""
    for cmd in backend.pattern.seq:
        if cmd[0] == "N":
            backend.add_nodes([cmd[1]])
        elif cmd[0] == "E":
            backend.entangle_nodes(cmd[1])
        elif cmd[0] == "M":
            backend.measure(cmd)
        elif cmd[0] == "X":
            backend.correct_byproduct(cmd)
        elif cmd[0] == "Z":
            backend.correct_byproduct(cmd)
        elif cmd[0] == "C":
            backend.apply_clifford(cmd)
        else:
            raise ValueError("invalid commands")
        if backend.pattern.seq[-1] == cmd:
            backend.finalize()


# %%
# Then, we define a random circuit generator:


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

test_cases = [(i, 1) for i in range(2, 22)]

pattern_time = []
circuit_time = []

# %%
# We then run simulations.
# First, we run the pattern simulations and circuit simulation.

for nqubit, depth in test_cases:
    circuit = simple_random_circuit(nqubit, depth)
    pattern = circuit.transpile()
    pattern.standardize()
    pattern.minimize_space()
    max_qubit_num = 20 if nqubit < 20 else 50
    backend = StatevectorBackend(pattern, max_qubit_num=max_qubit_num)
    print(f"max space for nqubit={nqubit} circuit is ", pattern.max_space())
    start = perf_counter()
    run(backend)
    end = perf_counter()
    print(f"nqubit: {nqubit}, depth: {depth}, time: {end - start}")
    pattern_time.append(end - start)
    start = perf_counter()
    circuit.simulate_statevector()
    end = perf_counter()
    circuit_time.append(end - start)


# %%
# Here we take benchmarking for MBQC simulation using `paddle_quantum`.


def simple_random_circuit_for_paddle_quantum(width, depth):
    r"""Generate a random circuit for paddle.

    This function generates a circuit with nqubit qubits and depth layers,
    having layers of CNOT and Rz gates with random placements.

    Parameters
    ----------
    width : int
        number of qubits
    depth : int
        number of layers

    Returns
    -------
    circuit : paddle_quantum.mbqc.qobject.Circuit
        generated circuit
    """
    qubit_index = [i for i in range(width)]
    circuit = PaddleCircuit(width)
    for _ in range(depth):
        np.random.shuffle(qubit_index)
        for j in range(len(qubit_index) // 2):
            circuit.cnot(which_qubits=[qubit_index[2 * j], qubit_index[2 * j + 1]])
        for j in range(len(qubit_index)):
            circuit.rz(which_qubit=qubit_index[j], theta=to_tensor(2 * np.pi * np.random.random(), dtype="float64"))
    return circuit


test_cases_for_paddle_quantum = [(i, 1) for i in range(2, 22)]
paddle_quantum_time = []

for width, depth in test_cases_for_paddle_quantum:
    cir = simple_random_circuit_for_paddle_quantum(width, depth)
    pat = PaddleTranspile(cir)
    mbqc = PaddleMBQC()
    mbqc.set_pattern(pat)
    start = perf_counter()
    mbqc.run_pattern()
    end = perf_counter()
    paddle_quantum_time.append(end - start)

    print(f"nqubit: {width}, depth: {depth}, time: {end - start}")

# %%
# Here we take benchmarking for MBQC simulation using `mentpy`.

DEPTH = 2  # For mentpy, due to its specific structure, we need to set depth to 2 instead of 1.
test_cases_for_mentpy = [(i, DEPTH) for i in range(2, 13)]
mentpy_time = []

for width, depth in test_cases_for_mentpy:
    grid_cluster = mp.templates.grid_cluster(n=width, m=depth)
    random_state = mp.utils.generate_haar_random_states(n_qubits=width)
    simulator = mp.PatternSimulator(grid_cluster, backend="numpy-sv", window_size=width)
    simulator.reset(input_state=random_state)
    angles = np.zeros(width)
    start = perf_counter()
    simulator.run(angles=angles)
    end = perf_counter()
    mentpy_time.append(end - start)

    print(f"nqubit: {width}, depth: {depth}, time: {end - start}")

# %%
# Lastly, we compare the simulation times.

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter([case[0] for case in test_cases], pattern_time, label="graphix pattern simulator with remove_qubit(new)")
ax.scatter(
    [case[0] for case in test_cases], circuit_time,
    label="direct statevector sim of original gate-based circuit (for reference)", marker="x"
)
ax.scatter(
    [case[0] for case in test_cases_for_paddle_quantum], paddle_quantum_time, label="paddle_quantum pattern simulator"
)
ax.scatter([case[0] for case in test_cases_for_mentpy], mentpy_time, label="mentpy pattern simulator")
ax.set(
    xlabel="nqubit",
    ylabel="time (s)",
    yscale="log",
    title="Time to simulate random circuits",
)
fig.legend(bbox_to_anchor=(0.65, 0.9))
fig.show()

# %%
