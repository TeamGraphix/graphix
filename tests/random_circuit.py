import numpy as np
from graphix.transpiler import Circuit


def first_rotation(circuit, nqubits):
    for qubit in range(nqubits):
        circuit.rx(qubit, np.random.rand())


def mid_rotation(circuit, nqubits):
    for qubit in range(nqubits):
        circuit.rx(qubit, np.random.rand())
        circuit.rz(qubit, np.random.rand())


def last_rotation(circuit, nqubits):
    for qubit in range(nqubits):
        circuit.rz(qubit, np.random.rand())


def entangler(circuit, pairs):
    for a, b in pairs:
        circuit.cnot(a, b)


def entangler_rzz(circuit, pairs):
    for a, b in pairs:
        circuit.rzz(a, b, np.random.rand())


def generate_gate(nqubits, depth, pairs, use_rzz=False):
    circuit = Circuit(nqubits)
    first_rotation(circuit, nqubits)
    entangler(circuit, pairs)
    for k in range(depth - 1):
        mid_rotation(circuit, nqubits)
        if use_rzz:
            entangler_rzz(circuit, pairs)
        else:
            entangler(circuit, pairs)
    last_rotation(circuit, nqubits)
    return circuit


def genpair(n_qubits, count):
    pairs = []
    for i in range(count):
        choice = [j for j in range(n_qubits)]
        x = np.random.choice(choice)
        choice.pop(x)
        y = np.random.choice(choice)
        pairs.append((x, y))
    return pairs


def get_rand_circuit(nqubits, depth, use_rzz=False):
    circuit = Circuit(nqubits)
    gate_choice = [0, 1, 2, 3, 4, 5, 6, 7]
    for i in range(depth):
        for j, k in genpair(nqubits, 2):
            circuit.cnot(j, k)
        if use_rzz:
            for j, k in genpair(nqubits, 1):
                circuit.rzz(j, k, np.pi / 4)
        for j in range(nqubits):
            k = np.random.choice(gate_choice)
            if k == 0:
                circuit.ry(j, np.pi / 4)
                pass
            elif k == 1:
                circuit.rz(j, -np.pi / 4)
                pass
            elif k == 2:
                circuit.rx(j, -np.pi / 4)
                pass
            elif k == 3:  # H
                circuit.h(j)
                pass
            elif k == 4:  # S
                circuit.s(j)
                pass
            elif k == 5:  # X
                circuit.x(j)
                pass
            elif k == 6:  # Z
                circuit.z(j)
                pass
            elif k == 7:  # Y
                circuit.y(j)
                pass
            else:
                pass
    return circuit
