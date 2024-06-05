"""
Usage and application of tensor network simulations
=====================================================

In this example, we will illustrate the usage and application of TN simulation of MBQC.

Firstly, let's import the relevant modules:
"""

# %%

from functools import reduce

import cotengra as ctg
import matplotlib.pyplot as plt
import numpy as np
import opt_einsum as oe
from scipy.optimize import minimize

from graphix import Circuit

# %%
# This application will be for the QAOA (Quantum Approximate Optimization Algorithm),
# which is an algorithm that can be used for example to solve combinatorical optimization problems.
# For this example a max cut problem will be solved on a star-like graph, so we can easier validate the results.

# %%
# Let's start with defining a helper function for buidling the circuit.

def ansatz(circuit, n, gamma, beta, iterations):
    for j in range(0, iterations):
        for i in range(1, n):
            circuit.cnot(i, 0)
            circuit.rz(0, gamma[j])
            circuit.cnot(i, 0)
        for i in range(0, n):
            circuit.rx(i, beta[j])

# %%
# Let's look at how the quantum circuit is going to be built.

n = 5  # This will result in a graph that would be too large for statevector simulation.
iterations = 2  # Define the number of iterations in the quantum circuit.
gamma = 2 * np.pi * np.random.rand(iterations)
beta = 2 * np.pi * np.random.rand(iterations)
# Define the circuit.
circuit = Circuit(n)
ansatz(circuit, n, gamma, beta, iterations)

# Transpile Circuit into pattern as it is needed for creating the TN.
pattern = circuit.transpile(opt=True).pattern
# Optimizing according to standardization algorithm of graphix.
pattern.standardize()
pattern.shift_signals()

# %%
# Print some properties of the graph.

nodes, edges = pattern.get_graph()
print(f"Number of nodes: {len(nodes)}")
print(f"Number of edges: {len(edges)}")

# %%
# Optimizing by performing Pauli measurements in the pattern using efficient stabilizer simulator.

pattern.perform_pauli_measurements(use_rustworkx=True)

# %%
# Simulate using the TN backend of graphix, which will return an MBQCTensorNet object.
# The graph_prep argument is optional,
# but with 'parallel' the TensorNetworkBackend will prepeare the graph state faster.

mbqc_tn = pattern.simulate_pattern(backend="tensornetwork", graph_prep="parallel")
sv = mbqc_tn.to_statevector().flatten()
print("Statevector after the simulation:", sv)

# %%
# Let's calculate the measuring probability corresponding to the first basis state.

value = mbqc_tn.get_basis_amplitude(0)
print("Probability for {} is {}".format(0, value))

# %%
# It is also possible to change the path contraction algorithm.
# Let's explore that too and define a custom optimizer for contraction, that we can use later.

opt = ctg.HyperOptimizer(
    minimize="combo",
    reconf_opts={},
    progbar=True,
)

# %%
# Let's also calculate the expectation value for the measurement in the computational basis.
# The expectation value can be optiained using a function of graphix.

pauli_z = np.array([[1, 0], [0, -1]])
identity = np.array([[1, 0], [0, 1]])
operator = reduce(np.kron, [pauli_z] * n)
# Use the defined optimizer by setting the 'optimize' parameter.
exp_val = mbqc_tn.expectation_value(operator, range(n), optimize=opt)
print("Expectation value for Z^n: ", exp_val)

# %%
# If we want to find the solution for our initial max-cut problem,
# then we must deploy a classical minimizer too for an apropriate cost function.
# Create a cost function using the elements of graphix, which were already discussed above.

def cost(params, n, ham, quantum_iter, slice_index, opt=None):
    circuit = Circuit(n)
    gamma = params[:slice_index]
    beta = params[slice_index:]
    ansatz(circuit, n, gamma, beta, quantum_iter)

    pattern = circuit.transpile(opt=True).pattern
    pattern.standardize()
    pattern.shift_signals()
    pattern.perform_pauli_measurements(use_rustworkx=True)
    mbqc_tn = pattern.simulate_pattern(backend="tensornetwork", graph_prep="parallel")
    exp_val = 0
    for op in ham:
        exp_val += np.real(mbqc_tn.expectation_value(op, range(n), optimize=opt))
    return exp_val

# %%
# We want to find the ground state energy for the Hamiltonian = \sum Z_k  + \sum Z_i Z_j  with i,j running over the edges.

ham = [reduce(np.kron, [pauli_z] * n)]
for i in range(1, n):
    op = [identity] * n
    op[0] = pauli_z
    op[i] = pauli_z
    op = reduce(np.kron, op)
    ham.append(op)

# Use yet again another optimizer for path contraction.
class MyOptimizer(oe.paths.PathOptimizer):
    def __call__(self, inputs, output, size_dict, memory_limit=None):
        return [(0, 1)] * (len(inputs) - 1)

opt = MyOptimizer()
# Define initial parameters, which will be optimized through running the algorithm.
params = 2 * np.pi * np.random.rand(len(gamma) + len(beta))
# Run the classical optimizer and simulate the quantum circuit with TN backend.
res = minimize(cost, params, args=(n, ham, iterations, len(gamma), opt), method="COBYLA")
print(res.message)

# %%
# Finally, run the circuit once again with the optimized parameters.

circuit = Circuit(n)
ansatz(circuit, n, res.x[: len(gamma)], res.x[len(gamma) :], iterations)
pattern = circuit.transpile(opt=True).pattern
pattern.standardize()
pattern.shift_signals()
mbqc_tn = pattern.simulate_pattern(backend="tensornetwork", graph_prep="parallel")

# %%
# Let's use the defined optimizer and find the most probable basis states.

max_prob = 0
most_prob_state = 0
bars = []
for i in range(0, 2**n):
    value = mbqc_tn.get_basis_amplitude(i)
    bars.append(value)

# %%
# Plot the output.

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(0, 2**n), bars, color="maroon", width=0.2)
ax.set_xticks(range(0, 2**n))
ax.set_xlabel("States")
ax.set_ylabel("Probabilites")
ax.set_title("Measurement probabilities using the optimized parameters")
plt.show()

# %%
# As we can see the most probable are 15 and 16 ( ``|11110>`` and ``|00001>`` because of bit ordering),
# which mean that splitting the graph so that node number 0 is in one set,
# and all other nodes in the other solves the max cut problem.
# This result is what we would expect from this star-like graph.

