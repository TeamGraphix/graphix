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
