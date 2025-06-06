from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pytest

from graphix.fundamentals import Plane
from graphix.generator import generate_from_graph
from graphix.random_objects import rand_gate

if TYPE_CHECKING:
    from numpy.random import Generator


class TestGenerator:
    def test_pattern_generation_determinism_flow(self, fx_rng: Generator) -> None:
        graph: nx.Graph[int] = nx.Graph([(0, 3), (1, 4), (2, 5), (1, 3), (2, 4), (3, 6), (4, 7), (5, 8)])
        inputs = {0, 1, 2}
        outputs = {6, 7, 8}
        angles = fx_rng.normal(size=6)
        results = []
        repeats = 3  # for testing the determinism of a pattern
        meas_planes = dict.fromkeys(range(6), Plane.XY)
        for _ in range(repeats):
            pattern = generate_from_graph(graph, angles, list(inputs), list(outputs), meas_planes=meas_planes)
            pattern.standardize()
            pattern.minimize_space()
            state = pattern.simulate_pattern(rng=fx_rng)
            results.append(state)
        combinations = [(0, 1), (0, 2), (1, 2)]
        for i, j in combinations:
            inner_product = np.dot(results[i].flatten(), results[j].flatten().conjugate())
            assert abs(inner_product) == pytest.approx(1)

    def test_pattern_generation_determinism_gflow(self, fx_rng: Generator) -> None:
        graph: nx.Graph[int] = nx.Graph([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (3, 6), (1, 6)])
        inputs = {1, 3, 5}
        outputs = {2, 4, 6}
        angles = fx_rng.normal(size=6)
        meas_planes = dict.fromkeys(range(1, 6), Plane.XY)
        results = []
        repeats = 3  # for testing the determinism of a pattern
        for _ in range(repeats):
            pattern = generate_from_graph(graph, angles, list(inputs), list(outputs), meas_planes=meas_planes)
            pattern.standardize()
            pattern.minimize_space()
            state = pattern.simulate_pattern(rng=fx_rng)
            results.append(state)
        combinations = [(0, 1), (0, 2), (1, 2)]
        for i, j in combinations:
            inner_product = np.dot(results[i].flatten(), results[j].flatten().conjugate())
            assert abs(inner_product) == pytest.approx(1)

    def test_pattern_generation_flow(self, fx_rng: Generator) -> None:
        nqubits = 3
        depth = 2
        pairs = [(0, 1), (1, 2)]
        circuit = rand_gate(nqubits, depth, pairs, fx_rng)
        # transpile into graph
        pattern = circuit.transpile().pattern
        pattern.standardize()
        pattern.shift_signals()
        # get the graph and generate pattern again with flow algorithm
        nodes, edges = pattern.get_graph()
        g: nx.Graph[int] = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        input_list = [0, 1, 2]
        angles: dict[int, float] = {}
        for cmd in pattern.get_measurement_commands():
            assert isinstance(cmd.angle, float)
            angles[cmd.node] = float(cmd.angle)
        meas_planes = pattern.get_meas_plane()
        pattern2 = generate_from_graph(g, angles, input_list, pattern.output_nodes, meas_planes)
        # check that the new one runs and returns correct result
        pattern2.standardize()
        pattern2.shift_signals()
        pattern2.minimize_space()
        state = circuit.simulate_statevector().statevec
        state_mbqc = pattern2.simulate_pattern(rng=fx_rng)
        assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)

    def test_pattern_generation_no_internal_nodes(self) -> None:
        g: nx.Graph[int] = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2)])
        pattern = generate_from_graph(g, {}, {0, 1, 2}, {0, 1, 2}, {})
        assert pattern.get_graph() == ([0, 1, 2], [(0, 1), (1, 2)])
