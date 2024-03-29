# manual test for mgraph.py
# This file will be replaced with unittest
# %%
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

NUMBA_NUM_THREADS = 1
from graphix.mgraph import COLOR_MAP, MGraph
from graphix.transpiler import Circuit

from tests.random_circuit import get_rand_circuit

import random

random.seed(42)

# %%
# ======================= #
# core test for MultiGraph
# each edge should be labeled with a unique number and Hadamard index
nodes = [(1, "XY", np.pi / 2), (2, "XY", np.pi / 3), (3, "XY", 0)]
edges = [
    (1, 2, {"hadamard": True}),
    (1, 2, {"hadamard": False}),
    (1, 2, {"hadamard": True}),
    (2, 3, {"hadamard": False}),
]
mgraph = MGraph()
for node, plane, angle in nodes:
    mgraph.add_node(node, plane, angle)
for edge in edges:
    mgraph.add_edge(edge[0], edge[1], **edge[2])
node_color, edge_color = mgraph.get_colors()
nx.draw(mgraph, node_color=node_color, edge_color=edge_color, with_labels=True)
print(mgraph.edges(data=True))
print(mgraph.edges([2], data=True))
print(mgraph.number_of_edges(1, 2))
print(mgraph[1])
# %%
# ======================= #
# fusion test
nodes = [
    # (1, "XY", np.pi / 2),
    (1, "XY", np.pi / 2),
    # (2, "XY", np.pi / 3),
    (2, "XY", np.pi / 3),
    (3, "XY", 0),
    (4, "XY", 0),
    (5, "XY", 0),
    (6, "XY", 0),
    (7, "XY", 0),
]
edges = [
    (1, 2, {"hadamard": False}),
    (1, 3, {"hadamard": True}),
    (1, 4, {"hadamard": True}),
    (1, 5, {"hadamard": True}),
    (2, 6, {"hadamard": True}),
    (2, 7, {"hadamard": True}),
]
mgraph = MGraph()
for node, plane, angle in nodes:
    mgraph.add_node(node, plane, angle)
for edge in edges:
    mgraph.add_edge(edge[0], edge[1], **edge[2])
node_color, edge_color = mgraph.get_colors()
nx.draw(mgraph, node_color=node_color, edge_color=edge_color, with_labels=True)
# %%
mgraph.fusion(1, 2, 8)
node_color, edge_color = mgraph.get_colors()
nx.draw(mgraph, node_color=node_color, edge_color=edge_color, with_labels=True)
# %%
# ======================= #
# hermite_conj
mgraph.hermite_conj(8)
node_color, edge_color = mgraph.get_colors()
nx.draw(mgraph, node_color=node_color, edge_color=edge_color, with_labels=True)
# %%
# identity

nodes = [
    (1, "XY", np.pi / 3),
    (2, "XY", 0),
    (3, "XY", np.pi / 7),
]

edges = [
    (1, 2, {"hadamard": False}),
    (2, 3, {"hadamard": False}),
]

mgraph = MGraph()
for node, plane, angle in nodes:
    mgraph.add_node(node, plane, angle)
for edge in edges:
    mgraph.add_edge(edge[0], edge[1], **edge[2])

node_color, edge_color = mgraph.get_colors()
nx.draw(mgraph, node_color=node_color, edge_color=edge_color, with_labels=True)
# %%
target = 2
mgraph = mgraph.identity(target)
node_color, edge_color = mgraph.get_colors()
nx.draw(mgraph, node_color=node_color, edge_color=edge_color, with_labels=True)

# %%
# ======================= #
# Hadamard cancel
mgraph = MGraph()

# 1, 4 are normal nodes; 2, 3 are Hadamard nodes
nodes = [1, 2, 3, 4]
edges = [
    (1, 2, {"hadamard": False}),
    (2, 3, {"hadamard": False}),
    (3, 4, {"hadamard": False}),
]

mgraph.add_node(1, "XY")
mgraph.add_node(4, "XY")

mgraph.add_hadamard(2)
mgraph.add_hadamard(3)

for edge in edges:
    mgraph.add_edge(edge[0], edge[1], **edge[2])

node_color, edge_color = mgraph.get_colors()
nx.draw(mgraph, node_color=node_color, edge_color=edge_color, with_labels=True)

# %%
mgraph.hadamard_cancel(2, 3, 5)
node_color, edge_color = mgraph.get_colors()
nx.draw(mgraph, node_color=node_color, edge_color=edge_color, with_labels=True)

# %%
mgraph.identity(5)
node_color, edge_color = mgraph.get_colors()
nx.draw(mgraph, node_color=node_color, edge_color=edge_color, with_labels=True)

# %%
# ======================= #
# pi transport

nodes = [
    (1, "XY", 0),
    (2, "YZ", np.pi),
    (3, "XY", 0),
    (4, "XY", 0),
    (5, "XY", 0),
    (6, "XY", 0),
]

edges = [
    (1, 2, {"hadamard": True}),
    (2, 3, {"hadamard": False}),
    (3, 4, {"hadamard": False}),
    (3, 5, {"hadamard": True}),
    (3, 6, {"hadamard": False}),
]

mgraph = MGraph()
for node, plane, angle in nodes:
    mgraph.add_node(node, plane, angle)
for edge in edges:
    mgraph.add_edge(edge[0], edge[1], **edge[2])
node_color, edge_color = mgraph.get_colors()
nx.draw(mgraph, node_color=node_color, edge_color=edge_color, with_labels=True)

# %%
new_indices = {4: 7, 5: 8, 6: 9}

mgraph.pi_transport(3, 2, new_indices)


node_color, edge_color = mgraph.get_colors()
nx.draw(mgraph, node_color=node_color, edge_color=edge_color, with_labels=True)

# %%
# ======================= #
# anchor divide

nodes = [
    (1, "YZ", 0),
    (2, "XY", 0),
    (3, "XY", 0),
    (4, "XY", 0),
    (5, "XY", 0),
]

edges = [
    (1, 2, {"hadamard": False}),
    (2, 3, {"hadamard": False}),
    (2, 4, {"hadamard": True}),
    (2, 5, {"hadamard": False}),
]

mgraph = MGraph()
for node, plane, angle in nodes:
    mgraph.add_node(node, plane, angle)
for edge in edges:
    mgraph.add_edge(edge[0], edge[1], **edge[2])
node_color, edge_color = mgraph.get_colors()
nx.draw(mgraph, node_color=node_color, edge_color=edge_color, with_labels=True)

# %%
new_indices = {3: 6, 4: 7, 5: 8}

mgraph.anchor_divide(2, 1, new_indices)

node_color, edge_color = mgraph.get_colors()
nx.draw(mgraph, node_color=node_color, edge_color=edge_color, with_labels=True)

# %%
# ======================= #
# bipartite rule
nodes = [
    (1, "XY", 0),
    (2, "XY", 0),
    (3, "YZ", 0),
    (4, "XY", 0),
    (5, "XY", 0),
    (6, "XY", 0),
]

edges = [
    (1, 3, {"hadamard": False}),
    (2, 3, {"hadamard": True}),
    (3, 4, {"hadamard": False}),
    (4, 5, {"hadamard": True}),
    (4, 6, {"hadamard": False}),
]
mgraph = MGraph()
for node, plane, angle in nodes:
    mgraph.add_node(node, plane, angle)
for edge in edges:
    mgraph.add_edge(edge[0], edge[1], **edge[2])
node_color, edge_color = mgraph.get_colors()
nx.draw(mgraph, node_color=node_color, edge_color=edge_color, with_labels=True)

# %%
new_indices = {"u1": 7, "u2": 8, "v1": 9, "v2": 10}
mgraph.bipartite(3, 4, new_indices)

node_color, edge_color = mgraph.get_colors()
nx.draw(mgraph, node_color=node_color, edge_color=edge_color, with_labels=True)

# %%
# check for local complementation
nodes = [
    (1, "XY", 0),
    (2, "XY", 0),
    (3, "YZ", 0),
    (4, "XY", 0),
]

edges = [
    (1, 3, {"hadamard": True}),
    (2, 3, {"hadamard": True}),
    (3, 4, {"hadamard": True}),
    (1, 2, {"hadamard": True}),
]
mgraph = MGraph()
for node, plane, angle in nodes:
    mgraph.add_node(node, plane, angle)
for edge in edges:
    mgraph.add_edge(edge[0], edge[1], **edge[2])
node_color, edge_color = mgraph.get_colors()
nx.draw(mgraph, node_color=node_color, edge_color=edge_color, with_labels=True)

# %%
mgraph.local_complementation(3)
node_color, edge_color = mgraph.get_colors()
nx.draw(mgraph, node_color=node_color, edge_color=edge_color, with_labels=True)
# %%
# local complementation test with circ structure
circ = get_rand_circuit(3, 2, seed=42)
mgraph = circ.to_mgraph()

node_color, edge_color = mgraph.get_colors()
nx.draw(mgraph, node_color=node_color, edge_color=edge_color, with_labels=True)
print(mgraph.flow)
# %%
mgraph.local_complementation(15)

node_color, edge_color = mgraph.get_colors()
nx.draw(mgraph, node_color=node_color, edge_color=edge_color, with_labels=True)
print(mgraph.flow)
