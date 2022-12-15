"""
Running larger QFT with MBQC on MPS simulator
"""

from graphix.transpiler import Circuit
import numpy as np


def cp(circuit, theta, control, target):
    circuit.rz(control, theta / 2)
    circuit.rz(target, theta / 2)
    circuit.cnot(control, target)
    circuit.rz(target, -1 * theta / 2)
    circuit.cnot(control, target)


def swap(circuit, a, b):
    circuit.cnot(a, b)
    circuit.cnot(b, a)
    circuit.cnot(a, b)


def qft_rotations(circuit, n):
    if n == 0:
        return circuit
    n -= 1
    circuit.h(n)
    for qubit in range(n):
        cp(circuit, np.pi / 2 ** (n - qubit), qubit, n)


def swap_registers(circuit, n):
    for qubit in range(n // 2):
        swap(circuit, qubit, n - qubit - 1)
    return circuit


def qft(circuit, n):
    for i in range(n):
        m = n - i
        qft_rotations(circuit, m)
    swap_registers(circuit, n)


n = 7
print("{}-qubit QFT".format(n))
circuit = Circuit(n)

# get |0> input states, from default |+> states
for i in range(n):
    circuit.h(i)
qft(circuit, n)

# standardize pattern
pattern = circuit.transpile()
pattern.standardize()
print("Simulating {}-node MBQC".format(pattern.max_space()))

# execute pattern on mps backend
mps = pattern.simulate_pattern(backend="mps")
value = mps.get_amplitude(0)
print("amplitude of |00000> is ", value)
print("1/2^n (true answer) is", 1 / 2 ** n)
