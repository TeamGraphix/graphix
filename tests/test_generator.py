from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pytest

import tests.random_circuit as rc
from graphix.generator import generate_from_graph

if TYPE_CHECKING:
    from numpy.random import Generator


class TestGenerator:
    def test_pattern_generation_determinism_flow(self, fx_rng: Generator) -> None:
        graph = nx.Graph([(0, 3), (1, 4), (2, 5), (1, 3), (2, 4), (3, 6), (4, 7), (5, 8)])
        inputs = {0, 1, 2}
        outputs = {6, 7, 8}
        angles = fx_rng.normal(size=6)
        results = []
        repeats = 3  # for testing the determinism of a pattern
        meas_planes = {i: "XY" for i in range(6)}
        for _ in range(repeats):
            pattern = generate_from_graph(graph, angles, list(inputs), list(outputs), meas_planes=meas_planes)
            pattern.standardize()
            pattern.minimize_space()
            state = pattern.simulate_pattern()
            results.append(state)
        combinations = [(0, 1), (0, 2), (1, 2)]
        for i, j in combinations:
            inner_product = np.dot(results[i].flatten(), results[j].flatten().conjugate())
            assert abs(inner_product) == pytest.approx(1)

    def test_pattern_generation_determinism_gflow(self, fx_rng: Generator) -> None:
        graph = nx.Graph([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (3, 6), (1, 6)])
        inputs = {1, 3, 5}
        outputs = {2, 4, 6}
        angles = fx_rng.normal(size=6)
        meas_planes = {i: "XY" for i in range(1, 6)}
        results = []
        repeats = 3  # for testing the determinism of a pattern
        for _ in range(repeats):
            pattern = generate_from_graph(graph, angles, list(inputs), list(outputs), meas_planes=meas_planes)
            pattern.standardize()
            pattern.minimize_space()
            state = pattern.simulate_pattern()
            results.append(state)
        combinations = [(0, 1), (0, 2), (1, 2)]
        for i, j in combinations:
            inner_product = np.dot(results[i].flatten(), results[j].flatten().conjugate())
            assert abs(inner_product) == pytest.approx(1)

    def test_pattern_generation_flow(self, fx_rng: Generator) -> None:
        nqubits = 3
        depth = 2
        pairs = [(0, 1), (1, 2)]
        circuit = rc.generate_gate(nqubits, depth, pairs, fx_rng)
        # transpile into graph
        pattern = circuit.transpile().pattern
        pattern.standardize()
        pattern.shift_signals()
        # get the graph and generate pattern again with flow algorithm
        nodes, edges = pattern.get_graph()
        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        input_list = [0, 1, 2]
        angles = {}
        for cmd in pattern.get_measurement_commands():
            angles[cmd[1]] = cmd[3]
        meas_planes = pattern.get_meas_plane()
        pattern2 = generate_from_graph(g, angles, input_list, pattern.output_nodes, meas_planes)
        # check that the new one runs and returns correct result
        pattern2.standardize()
        pattern2.shift_signals()
        pattern2.minimize_space()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern2.simulate_pattern()
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)
