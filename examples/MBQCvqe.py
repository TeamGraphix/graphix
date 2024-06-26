"""
Variational Quantum Eigensolver (VQE) with Measurement-Based Quantum Computing (MBQC)
====================================================================================

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

import networkx as nx
import numpy as np
from scipy.optimize import minimize

from graphix import Circuit
from graphix.simulator import PatternSimulator


# %%
# Define the Hamiltonian for the VQE problem (Example: H = Z0Z1 + X0 + X1)
def create_hamiltonian():
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    H = np.kron(Z, Z) + np.kron(X, np.eye(2)) + np.kron(np.eye(2), X)
    return H


# %%
# Function to build the VQE circuit
def build_vqe_circuit(n_qubits, params):
    circuit = Circuit(n_qubits)
    for i in range(n_qubits):
        circuit.rx(i, params[i])
        circuit.ry(i, params[i + n_qubits])
        circuit.rz(i, params[i + 2 * n_qubits])
    for i in range(n_qubits - 1):
        circuit.cnot(i, i + 1)
    return circuit


# %%
class MBQCVQE:
    def __init__(self, n_qubits, hamiltonian):
        self.n_qubits = n_qubits
        self.hamiltonian = hamiltonian

    # %%
    # Function to build the MBQC pattern
    def build_mbqc_pattern(self, params):
        circuit = build_vqe_circuit(self.n_qubits, params)
        pattern = circuit.transpile().pattern
        pattern.standardize()
        pattern.shift_signals()
        return pattern

    # %%
    # Function to simulate the MBQC circuit
    def simulate_mbqc(self, params, backend="tensornetwork"):
        pattern = self.build_mbqc_pattern(params)
        pattern.perform_pauli_measurements()  # Perform Pauli measurements
        simulator = PatternSimulator(pattern, backend=backend)
        if backend == "tensornetwork":
            tn = simulator.run()  # Simulate the MBQC circuit using tensor network
            tn.default_output_nodes = pattern.output_nodes  # Set the default_output_nodes attribute
            if tn.default_output_nodes is None:
                raise ValueError("Output nodes are not set for tensor network simulation.")
            return tn
        else:
            out_state = simulator.run()  # Simulate the MBQC circuit using other backends
            return out_state

    # %%
    # Function to compute the energy
    def compute_energy(self, params):
        # Simulate the MBQC circuit using tensor network backend
        tn = self.simulate_mbqc(params, backend="tensornetwork")
        # Compute the expectation value using MBQCTensornet.expectation_value
        energy = tn.expectation_value(self.hamiltonian, qubit_indices=range(self.n_qubits))
        return energy


# %%
# Set parameters for VQE
n_qubits = 2
hamiltonian = create_hamiltonian()

# %%
# Instantiate the MBQCVQE class
mbqc_vqe = MBQCVQE(n_qubits, hamiltonian)


# %%
# Define the cost function
def cost_function(params):
    return mbqc_vqe.compute_energy(params)


# %%
# Random initial parameters
initial_params = np.random.rand(n_qubits * 3)

# %%
# Perform the optimization using COBYLA
result = minimize(cost_function, initial_params, method="COBYLA", options={"maxiter": 100})

print(f"Optimized parameters: {result.x}")
print(f"Optimized energy: {result.fun}")

# %%
# Compare with the analytical solution
analytical_solution = -np.sqrt(2) - 1
print(f"Analytical solution: {analytical_solution}")
