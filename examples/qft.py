"""
Graphix example with 3-qubit QFT circuit and comparison with statevec sim
"""

from graphix.transpiler import Circuit
import numpy as np


def cp(circuit, theta, control, target):
    """Controlled rotation gate, decomposed
    """
    circuit.rz(control, theta / 2)
    circuit.rz(target, theta / 2)
    circuit.cnot(control, target)
    circuit.rz(target, -1 * theta / 2)
    circuit.cnot(control, target)


def swap(circuit, a, b):
    """swap gate, decomposed
    """
    circuit.cnot(a, b)
    circuit.cnot(b, a)
    circuit.cnot(a, b)


circuit = Circuit(3)

# prepare all states in |0>, not |+> (default for graph state)
circuit.h(0)
circuit.h(1)
circuit.h(2)

circuit.x(1)
circuit.x(2)

# QFT
circuit.h(2)
cp(circuit, np.pi / 4, 0, 2)
cp(circuit, np.pi / 2, 1, 2)
circuit.h(1)
cp(circuit, np.pi / 2, 0, 1)
circuit.h(0)
swap(circuit, 0, 2)

# run with MBQC simulator
pat = circuit.transpile()
pat.standardize()
pat.shift_signals()
pat.perform_pauli_measurements()
pat.minimize_space()
out_state = pat.simulate_pattern()

state = circuit.simulate_statevector()
print("overlap of states: ", np.abs(np.dot(state.psi.flatten().conjugate(), out_state.psi.flatten())))
