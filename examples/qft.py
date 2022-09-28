from graphix.simulator import Simulator
from graphix.transpiler import Circuit
import numpy as np

# 2-qubit QFT
# more description:https://qiskit.org/textbook/ch-algorithms/quantum-fourier-transform.html


def cp(circuit, theta, control, target):
    """Controlled rotation gate, decomposed
    """
    circuit.rz(control, theta / 2)
    circuit.rz(target, theta / 2)
    circuit.cnot(control, target)
    circuit.rz(target, -1 * theta / 2)
    circuit.cnot(control, target)


def swap(circuit, a, b):
    """swap gate
    see https://qiskit.org/textbook/ch-gates/more-circuit-identities.html#2.-Swapping-Qubits-
    """
    circuit.cnot(a, b)
    circuit.cnot(b, a)
    circuit.cnot(a, b)


circuit = Circuit(2)

# prepare all states in |0>, not |+> (default for graph state)
circuit.h(0)
circuit.h(1)

# input is b11=3
circuit.x(0)
circuit.x(1)

# QFT
circuit.h(1)
cp(circuit, np.pi / 2, 0, 1)
circuit.h(0)
swap(circuit, 0, 1)
circuit.sort_outputs()


# run with MBQC simulator
simulator = Simulator(circuit)
simulator.measure_pauli()
out_state = simulator.simulate_mbqc()

state = circuit.simulate_statevector()
print('overlap of states: ', np.abs(np.dot(state.data.conjugate(), out_state.data)))
