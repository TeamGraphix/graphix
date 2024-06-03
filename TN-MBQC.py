import networkx as nx
import numpy as np

from graphix import Circuit
from graphix.simulator import PatternSimulator

# Define the Hamiltonian for the VQE problem (Example: H = Z0Z1 + X0 + X1)
def create_hamiltonian():
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    H = np.kron(Z, Z) + np.kron(X, np.eye(2)) + np.kron(np.eye(2), X)
    return H

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

# Define the class for VQE with MBQC
class MBQCVQE:
    def __init__(self, n_qubits, hamiltonian):
        self.n_qubits = n_qubits
        self.hamiltonian = hamiltonian

    def build_mbqc_pattern(self, params):
        circuit = build_vqe_circuit(self.n_qubits, params)
        pattern = circuit.transpile().pattern
        pattern.standardize()
        pattern.shift_signals()
        return pattern

    def simulate_mbqc(self, params, backend="statevector"):
        pattern = self.build_mbqc_pattern(params)
        pattern.perform_pauli_measurements()  # Perform Pauli measurements
        simulator = PatternSimulator(pattern, backend=backend)
        if backend == "tensornetwork":
            tn = simulator.run()  # Simulate the MBQC circuit using tensor network
            return tn
        else:
            out_state = simulator.run()  # Simulate the MBQC circuit using other backends
            return out_state

    def compute_energy(self, params):
        # Simulate the statevector using MBQC
        out_state = self.simulate_mbqc(params)
        statevec = out_state.psi.flatten()
        # Compute the expectation value <psi|H|psi>
        energy = np.real(np.dot(statevec.conjugate().T, np.dot(self.hamiltonian, statevec)))
        return energy

# Set parameters for VQE
n_qubits = 2
hamiltonian = create_hamiltonian()
params = np.random.rand(n_qubits * 3)  # Random initial parameters

# Instantiate the MBQCVQE class
mbqc_vqe = MBQCVQE(n_qubits, hamiltonian)

# Simulate the VQE circuit using MBQC with tensor network backend
tn = mbqc_vqe.simulate_mbqc(params, backend="tensornetwork")

# Compute the energy
energy = mbqc_vqe.compute_energy(params)
print(f"Computed energy: {energy}")

# Visualize the MBQC pattern
pattern = mbqc_vqe.build_mbqc_pattern(params)
pattern.draw_graph(flow_from_pattern=False)

# Simulate the MBQC pattern with tensor network backend
tn_pattern = pattern.simulate_pattern(backend="tensornetwork")
print(f"The amplitude of |00...0>: {tn_pattern.get_basis_amplitude(0)}")
print(f"The amplitude of |11...1>: {tn_pattern.get_basis_amplitude(2**n_qubits - 1)}")

# Simulate the VQE circuit and compute the overlap of states
out_state = pattern.simulate_pattern()
state = Circuit(n_qubits).simulate_statevector().statevec
print("Overlap of states: ", np.abs(np.dot(state.psi.flatten().conjugate(), out_state.psi.flatten())))


#  Extended QAOA using MBQC Class 


import networkx as nx
import numpy as np

from graphix import Circuit
from graphix.simulator import PatternSimulator
from graphix.channels import KrausChannel, dephasing_channel
from graphix.noise_models.noise_model import NoiseModel

class MBQCQAOA:
    def __init__(self, n, xi, theta):
        self.n = n
        self.xi = xi
        self.theta = theta

    def build_mbqc_pattern(self):
        circuit = build_qaoa_circuit(self.n, self.xi, self.theta)
        pattern = circuit.transpile().pattern
        pattern.standardize()
        pattern.shift_signals()
        return pattern

    def simulate_mbqc(self, backend="statevector"):
        pattern = self.build_mbqc_pattern()
        pattern.perform_pauli_measurements()  # Perform Pauli measurements
        simulator = PatternSimulator(pattern, backend=backend)
        if backend == "tensornetwork":
            tn = simulator.run()  # Simulate the MBQC circuit using tensor network
            return tn
        else:
            out_state = simulator.run()  # Simulate the MBQC circuit using other backends
            return out_state

# Set parameters for QAOA
n = 4
xi = np.random.rand(6)
theta = np.random.rand(4)

# Instantiate the MBQCQAOA class
mbqc_qaoa = MBQCQAOA(n, xi, theta)

# Simulate the QAOA circuit using MBQC with tensor network backend
tn = mbqc_qaoa.simulate_mbqc(backend="tensornetwork")
print(f"The amplitude of |00>: {tn.get_basis_amplitude(0)}")
print(f"The amplitude of |11>: {tn.get_basis_amplitude(3)}")
pattern = circuit.transpile().pattern
pattern.print_pattern(lim=50)
pattern.perform_pauli_measurements()
pattern.draw_graph(flow_from_pattern=False, node_distance=(1, 0.6))


tn = pattern.simulate_pattern(backend="tensornetwork")
print(f"The amplitude of |00...0>: {tn.get_basis_amplitude(0)}")
print(f"The amplitude of |00...0>: {tn.get_basis_amplitude(2**n-1)}")
