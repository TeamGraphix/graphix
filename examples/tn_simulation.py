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
import networkx as nx
import numpy as np
import opt_einsum as oe
import quimb.tensor as qtn
from scipy.optimize import minimize

from graphix import Circuit
from graphix.gflow import find_gflow
from graphix.visualization import GraphVisualizer

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
# Let's explore what is really happening, how the TN is being constructed.
# The tensor network is created using the graph structure, so from the list of nodes as well as the edges.
# The graph must be preprocessed before the constraction of the TN itself.

# %%
# The goal is to represent quantum states, so for every node a list is created, which stores the index labels for each dimension for that given node.
# Its length will be one larger than the number of edges of the node.
# This is due to the fact that the first entry of the list represents the "dangling" index, which corresponds to the physical index of the qubit (i.e., the index that represents the local Hilbert space of the qubit).
# The following entries in the list are then correspond to neighbouring tensors, and can be contracted with them.
# For additional details and visualization visit: https://quimb.readthedocs.io/en/latest/tensor-1d.html.

# %%
# Let's take a closer look at an MPS tensor (left plot) and an MPS tensor network that consists of two MPS tensors (right plot).
# By the network on the right the middle index is shared between the two tensors, essentially allowing for contraction between them by summing over it.

t = qtn.rand_tensor([2], "a")
fig, ax = plt.subplots(1, 2)
t.draw(
    ax=ax[0],
    title="MPS tensor",
    legend=False,
)
t1 = qtn.rand_tensor([2, 2], ["a", "b"])
t2 = qtn.rand_tensor([1, 2], ["b", "c"])
t1.add_tag("T1")
t2.add_tag("T2")
t = qtn.TensorNetwork([t1, t2])
t.draw(ax=ax[1], title="MPS", legend=False, color=["T1", "T2"])
plt.show()

# %%
# Additionally, the type of edges are also stored, in a binary valued list for each node.
# These are used to construct the tensor itself.
# From each node in the graph a tensor is constructed, which has a dimension that is exactly one larger than its neighbour count.
# The tensor is described using two outer products, for which the list from above is used, that describes the edges for every node.
# For additional information on TN construction please refer to: https://journals.aps.org/pra/abstract/10.1103/PhysRevA.76.052315 .
# Section III A provides further information on Matrix Porduct States and section III C gives an example using a 1-D cluster state.
# In section IV novel resource states are explored, where parts A, B can be used for getting a deeper understanding.

# %%
# Let's also plot the resulting tensor network (notice that there are five dangling edges, which is exactly the number of qubits that were defined in the quantum circuit).
# Here open means that the node has a dangling index, but the "dangling" edge itself is not drawn, rather the tensors are color coded.

fig, ax = plt.subplots(figsize=(13, 10))
color = ["Open", "Close"]

# Rebuilding the graph to be visualizable
# Ignoring dangling edges when plotting, but coloring according to them
ind_map, tag_map, default_output_nodes = mbqc_tn.ind_map, mbqc_tn.tag_map, mbqc_tn.default_output_nodes
nodes = set()
nodes.update(list(tag_map["Open"]))  # The node has "dangling" index
nodes.update(list(tag_map["Close"]))
edges = []
for i in ind_map.items():
    if len(i[1]) < 2:
        continue
    edges.append(list(i[1]))
input_nodes = []
for node in nodes:
    candidate = True
    for edge in edges:
        if edge[1] == node:
            candidate = False
            continue
    if candidate:
        input_nodes.append(node)
for i in range(0, len(default_output_nodes)):
    default_output_nodes[i] = str(default_output_nodes[i])
out_nodes = []
for i in tag_map.items():
    if i[0] in default_output_nodes:
        out_nodes.append(list(i[-1])[-1])
meas_planes = dict()
for c in ["M", "Z", "C", "X"]:
    if c in tag_map.keys():
        for i in list(tag_map[c]):
            meas_planes[i] = c
for i in nodes:
    if i not in meas_planes.keys():
        meas_planes[i] = ""

G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)
f, l_k = find_gflow(G, input=set(input_nodes), output=set(out_nodes), meas_planes=meas_planes)
pos = GraphVisualizer(G, input_nodes, out_nodes).get_pos_from_gflow(g=f, l_k=l_k)

mbqc_tn.draw(
    ax=ax,
    color=color,
    show_tags=False,
    show_inds=False,
    fix=pos,
    iterations=100,
    k=0.01,
)
plt.show()

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
# We want to find the ground state energy for the Hamiltonian :math:`\hat{H} = \sum \hat{Z}_k + \sum \hat{Z}_i \hat{Z}_j` with i,j running over the edges.

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

# %%
# The following illustration shows the starting graph on the left,
# and the graph with the resulting sets found on the right, where the nodes with different colours belong to different groups.

fig, ax = plt.subplots(ncols=2, figsize=(8, 6))
ax = ax.flatten()
g = nx.Graph()
for i in range(1, n):
    g.add_edge(0, i)
color = ["blue"] * n
color[0] = "red"
nx.draw(g, ax=ax[0], with_labels=True)
nx.draw(g, ax=ax[1], node_color=color, with_labels=True)
plt.show()
