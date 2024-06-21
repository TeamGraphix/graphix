import random
import unittest
from collections import defaultdict
from copy import deepcopy

import matplotlib.pyplot as plt
import networkx as nx

import graphix.client
import graphix.gflow
import graphix.pauli
import graphix.visualization
import tests.random_circuit as rc
from graphix.client import TrappifiedCanvas
from graphix.sim.statevec import StatevectorBackend
from graphix.states import BasicStates


class TestVBQC(unittest.TestCase):

    def test_trap_delegated(self):
        nqubits = 2
        depth = 2
        circuit = rc.get_rand_circuit(nqubits, depth)
        pattern = circuit.transpile()
        pattern.standardize()
        states = [BasicStates.PLUS for _ in pattern.input_nodes]
        secrets = graphix.client.Secrets(r=True, a=True, theta=True)
        client = graphix.client.Client(pattern=pattern, input_state=states, secrets=secrets)
        test_runs, _ = client.create_test_runs()
        for run in test_runs:
            backend = StatevectorBackend()
            _, trap_outcomes = client.delegate_test_run(backend=backend, run=run)
            assert trap_outcomes == [0 for _ in run.traps_list]

    def test_stabilizer(self):
        nqubits = 2
        depth = 2
        circuit = rc.get_rand_circuit(nqubits, depth)
        pattern = circuit.transpile()
        pattern.standardize()
        nodes, edges = pattern.get_graph()[0], pattern.get_graph()[1]
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        import random

        k = random.randint(0, len(graph.nodes) - 1)
        nodes_sample = random.sample(list(graph.nodes), k)
        # nodes_sample=[0]

        stabilizer = graphix.client.Stabilizer(graph=graph, nodes=nodes_sample)

        expected_stabilizer = [graphix.pauli.I for _ in graph.nodes]
        for node in nodes_sample:
            expected_stabilizer[node] @= graphix.pauli.X
            for n in graph.neighbors(node):
                expected_stabilizer[n] @= graphix.pauli.Z

        assert expected_stabilizer == stabilizer.chain
