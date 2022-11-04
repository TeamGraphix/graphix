from graphix.transpiler import Circuit
import qiskit.quantum_info as qi
import numpy as np

# balanced oracle for 4 qubits
# more description: https://qiskit.org/textbook/ch-algorithms/deutsch-jozsa.html

circuit = Circuit(4)

# prepare all states in |0>, not |+> (default for graph state)
circuit.h(0)
circuit.h(1)
circuit.h(2)
circuit.h(3)

# initialization
circuit.h(0)
circuit.h(1)
circuit.h(2)

# ancilla
circuit.x(3)
circuit.h(3)

# balanced oracle
circuit.x(0)
circuit.x(2)

circuit.cnot(0, 3)
circuit.cnot(1, 3)
circuit.cnot(2, 3)

circuit.x(0)
circuit.x(2)

# finally, hadamard
circuit.h(0)
circuit.h(1)
circuit.h(2)

# run with MBQC simulator
pat = circuit.transpile()
pat.standardize()
pat.shift_signals()
pat.perform_pauli_measurements()
pat.minimize_space()
out_state = pat.simulate_pattern()
out_state = qi.partial_trace(out_state, [3]).to_statevector()
print('MBQC sampling result: ', out_state.sample_counts(1000))

# statevector sim
state = circuit.simulate_statevector()
state = qi.partial_trace(state, [3]).to_statevector()
print('desired sampling result: ', state.sample_counts(1000))
print('overlap of states: ', np.abs(np.dot(state.data.conjugate(), out_state.data)))
