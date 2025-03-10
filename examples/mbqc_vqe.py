"""
Variational Quantum Eigensolver (VQE) with Measurement-Based Quantum Computing (MBQC)
=====================================================================================

In this example, we solve a simple VQE problem using a measurement-based quantum
computing (MBQC) approach. The Hamiltonian for the system is given by:

.. math::

    H = Z_0 Z_1 + X_0 + X_1

where :math:`Z` and :math:`X` are the Pauli-Z and Pauli-X matrices, respectively.

This Hamiltonian corresponds to a simple model system often used in quantum computing
to demonstrate algorithms like VQE. The goal is to find the ground state energy of this
Hamiltonian.

We will build a parameterized quantum circuit and optimize its parameters to minimize
the expectation value of the Hamiltonian, effectively finding the ground state energy.
"""

import itertools
import sys
from collections.abc import Iterable
from timeit import timeit

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from graphix import Circuit
from graphix.parameter import Placeholder
from graphix.pattern import Pattern
from graphix.simulator import PatternSimulator
from graphix.transpiler import Angle

Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])


# %%
# Define the Hamiltonian for the VQE problem (Example: H = Z0Z1 + X0 + X1)
def create_hamiltonian() -> npt.NDArray[np.complex128]:
    return np.kron(Z, Z) + np.kron(X, np.eye(2)) + np.kron(np.eye(2), X)


if sys.version_info >= (3, 12):
    batched = itertools.batched
else:
    # From https://docs.python.org/3/library/itertools.html#itertools.batched
    def batched(iterable, n):
        # batched('ABCDEFG', 3) → ABC DEF G
        if n < 1:
            raise ValueError("n must be at least one")
        iterator = iter(iterable)
        while batch := tuple(itertools.islice(iterator, n)):
            yield batch


# %%
# Function to build the VQE circuit
def build_vqe_circuit(n_qubits: int, params: Iterable[Angle]) -> Circuit:
    circuit = Circuit(n_qubits)
    for i, (x, y, z) in enumerate(batched(params, n=3)):
        circuit.rx(i, x)
        circuit.ry(i, y)
        circuit.rz(i, z)
    for i in range(n_qubits - 1):
        circuit.cnot(i, i + 1)
    return circuit


# %%
class MBQCVQE:
    def __init__(self, n_qubits: int, hamiltonian: npt.NDArray):
        self.n_qubits = n_qubits
        self.hamiltonian = hamiltonian

    # %%
    # Function to build the MBQC pattern
    def build_mbqc_pattern(self, params: Iterable[Angle]) -> Pattern:
        circuit = build_vqe_circuit(self.n_qubits, params)
        pattern = circuit.transpile().pattern
        pattern.standardize()
        pattern.shift_signals()
        pattern.perform_pauli_measurements()  # Perform Pauli measurements
        return pattern

    # %%
    # Function to simulate the MBQC circuit
    def simulate_mbqc(self, params: Iterable[float], backend="tensornetwork"):
        pattern = self.build_mbqc_pattern(params)
        simulator = PatternSimulator(pattern, backend=backend)
        if backend == "tensornetwork":
            simulator.run()  # Simulate the MBQC circuit using tensor network
            tn = simulator.backend.state
            tn.default_output_nodes = pattern.output_nodes  # Set the default_output_nodes attribute
            if tn.default_output_nodes is None:
                raise ValueError("Output nodes are not set for tensor network simulation.")
            return tn
        return simulator.run()  # Simulate the MBQC circuit using other backends

    # %%
    # Function to compute the energy
    def compute_energy(self, params: Iterable[float]):
        # Simulate the MBQC circuit using tensor network backend
        tn = self.simulate_mbqc(params, backend="tensornetwork")
        # Compute the expectation value using MBQCTensornet.expectation_value
        return tn.expectation_value(self.hamiltonian, qubit_indices=range(self.n_qubits))


class MBQCVQEWithPlaceholders(MBQCVQE):
    def __init__(self, n_qubits: int, hamiltonian) -> None:
        super().__init__(n_qubits, hamiltonian)
        self.placeholders = tuple(Placeholder(f"{r}[{q}]") for q in range(n_qubits) for r in ("X", "Y", "Z"))
        self.pattern = super().build_mbqc_pattern(self.placeholders)

    def build_mbqc_pattern(self, params):
        return self.pattern.xreplace(dict(zip(self.placeholders, params)))


# %%
# Set parameters for VQE
n_qubits = 2
hamiltonian = create_hamiltonian()

# %%
# Instantiate the MBQCVQE class
mbqc_vqe = MBQCVQEWithPlaceholders(n_qubits, hamiltonian)


# %%
# Define the cost function
def cost_function(params):
    return mbqc_vqe.compute_energy(params)


# %%
# Random initial parameters
rng = np.random.default_rng()
initial_params = rng.random(n_qubits * 3)


# %%
# Perform the optimization using COBYLA
def compute():
    return minimize(cost_function, initial_params, method="COBYLA", options={"maxiter": 100})


result = compute()

print(f"Optimized parameters: {result.x}")
print(f"Optimized energy: {result.fun}")

# %%
# Compare with the analytical solution
analytical_solution = -np.sqrt(2) - 1
print(f"Analytical solution: {analytical_solution}")

# %%
# Compare performances between using parameterized circuits (with placeholders) or not

mbqc_vqe = MBQCVQEWithPlaceholders(n_qubits, hamiltonian)
time_with_placeholders = timeit(compute, number=2)
print(f"Time with placeholders: {time_with_placeholders}")

mbqc_vqe = MBQCVQE(n_qubits, hamiltonian)
time_without_placeholders = timeit(compute, number=2)
print(f"Time without placeholders: {time_without_placeholders}")
