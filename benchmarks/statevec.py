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
from mentpy import MBQCircuit as MentpyCircuit
import mentpy as mp

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

pattern_time = []
circuit_time = []

# %%
# We then run simulations.
# First, we run the pattern simulations and circuit simulation.

for nqubit in test_cases:
    circuit = simple_random_circuit(nqubit, DEPTH)
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


def simple_random_circuit_for_paddle_quantum(width, depth):
    """Generate a random circuit for paddle.

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


test_cases_for_paddle_quantum = [i for i in range(2, 22)]
paddle_quantum_time = []

for width in test_cases_for_paddle_quantum:
    cir = simple_random_circuit_for_paddle_quantum(width, DEPTH)
    pat = PaddleTranspile(cir)
    mbqc = PaddleMBQC()
    mbqc.set_pattern(pat)
    start = perf_counter()
    mbqc.run_pattern()
    end = perf_counter()
    paddle_quantum_time.append(end - start)

    print(f"nqubit: {width}, depth: {DEPTH}, time: {end - start}")

# %%
# Here we take benchmarking for MBQC simulation using `mentpy`.


def _cnot_dummy(gs: mp.GraphState, control_node: int, target_node: int, ancilla: list, measurements: dict) -> int:
    """approximate MBQC commands for CNOT gate.
    NOTE: This function is dummy. It aims to provide a pattern which is approximately equivalent to the CNOT gate.
    Since mentpy does not support byproduct correction, we just put a X measurement on the ancilla node as an
    alternative.

    Parameters
    ----------
        gs : mp.GraphState
            graph state
        control_node : int
            input node on graph
        target_node : int
            target node on graph
        ancilla : list
            ancilla node indices to be added to graph
        measurements : dict
            measurement operators

    Returns
    -------
        output_node : int
            output node
    """
    assert len(ancilla) == 2
    gs.add_nodes_from(ancilla)
    gs.add_edges_from([(target_node, ancilla[0]), (control_node, ancilla[0]), (ancilla[0], ancilla[1])])
    measurements[target_node] = mp.Ment(0, "XY")
    measurements[ancilla[0]] = mp.Ment(0, "XY")
    # NOTE: dummy measurement
    measurements[ancilla[1]] = mp.Ment("X")
    return control_node, ancilla[1]


def _rz_dummy(gs: mp.GraphState, input_node: int, ancilla: list, measurements: dict, angle: float) -> int:
    """approximate MBQC commands for Z rotation gate.
    NOTE: This function is dummy. It aims to provide a pattern which is approximately equivalent to the Z rotation
    gate. Since mentpy does not support byproduct correction, we just put a X measurement on the ancilla node as
    an alternative.

    Parameters
    ----------
        gs : mp.GraphState
            graph state
        input_node : int
            input node on graph
        ancilla : list
            ancilla node indices to be added to graph
        measurements : dict
            measurement operators
        angle : float
            rotation angle

    Returns
    -------
        output_node : int
            output node
    """
    assert len(ancilla) == 2
    gs.add_nodes_from(ancilla)
    gs.add_edges_from([(input_node, ancilla[0]), (ancilla[0], ancilla[1])])
    measurements[input_node] = mp.Ment(-angle / np.pi, "XY")
    measurements[ancilla[0]] = mp.Ment(0, "XY")
    # NOTE: dummy measurement
    measurements[ancilla[1]] = mp.Ment("X")
    return ancilla[1]


def translate_graphix_rc_into_mentpy_circuit(circuit: Circuit) -> MentpyCircuit:
    """Translate graphix circuit into mentpy circuit.

    Parameters
    ----------
        circuit : Circuit
            graphix circuit

    Returns
    -------
        mentpy_circ : MentpyCircuit
            mentpy circuit
    """
    gs = mp.GraphState()
    gs.add_nodes_from([i for i in range(circuit.width)])
    measurements = {}
    Nnode = circuit.width
    input = [j for j in range(circuit.width)]
    out = [j for j in range(circuit.width)]
    for instr in circuit.instruction:
        if instr[0] == "CNOT":
            ancilla = [Nnode, Nnode + 1]
            out[instr[1][0]], out[instr[1][1]] = _cnot_dummy(
                gs, out[instr[1][0]], out[instr[1][1]], ancilla, measurements
            )
            Nnode += 2
        elif instr[0] == "Rz":
            ancilla = [Nnode, Nnode + 1]
            out[instr[1]] = _rz_dummy(gs, out[instr[1]], ancilla, measurements, instr[2])
            Nnode += 2

    mentpy_circ = MentpyCircuit(gs, input_nodes=input, output_nodes=out)
    for node, op in measurements.items():
        if node in mentpy_circ._trainable_nodes:
            mentpy_circ[node] = op
    return mentpy_circ


test_cases_for_mentpy = [i for i in range(2, 13)]
mentpy_time = []

for width in test_cases_for_mentpy:
    graphix_circuit = simple_random_circuit(width, DEPTH)
    pattern = graphix_circuit.transpile()
    mentpy_circuit = translate_graphix_rc_into_mentpy_circuit(graphix_circuit)
    pattern.draw_graph()
    mp.draw(mentpy_circuit)
    simulator = mp.PatternSimulator(mentpy_circuit, backend="numpy-sv")
    angles = np.zeros(len(mentpy_circuit.trainable_nodes))
    start = perf_counter()
    simulator.run(angles=angles)
    end = perf_counter()
    mentpy_time.append(end - start)

    print(f"nqubit: {width}, depth: {DEPTH}, time: {end - start}")


# %%
# Lastly, we compare the simulation times.

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(
    test_cases, circuit_time, label="direct statevector sim of original gate-based circuit (for reference)", marker="x"
)
ax.scatter(test_cases, pattern_time, label="graphix pattern simulator")
ax.scatter(test_cases_for_paddle_quantum, paddle_quantum_time, label="paddle_quantum pattern simulator")
ax.scatter(test_cases_for_mentpy, mentpy_time, label="mentpy pattern simulator")
ax.set(
    xlabel="nqubit",
    ylabel="time (s)",
    yscale="log",
    title="Time to simulate random circuits",
)
fig.legend(bbox_to_anchor=(0.85, 0.9))
fig.show()

# %%
