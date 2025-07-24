from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pytest

from graphix.fundamentals import Plane
from graphix.generator import generate_from_graph
from graphix.gflow import find_gflow, find_pauliflow, pauliflow_from_pattern
from graphix.measurements import Measurement
from graphix.opengraph import OpenGraph
from graphix.random_objects import rand_gate

if TYPE_CHECKING:
    from numpy.random import Generator


class TestGenerator:
    def get_graph_pflow(self, fx_rng: Generator) -> OpenGraph:
        """Create a graph which has pflow but no gflow.

        Parameters
        ----------
        fx_rng : :class:`numpy.random.Generator`
            See graphix.tests.conftest.py

        Returns
        -------
        OpenGraph: :class:`graphix.opengraph.OpenGraph`
        """
        graph: nx.Graph[int] = nx.Graph(
            [(0, 2), (1, 4), (2, 3), (3, 4), (2, 5), (3, 6), (4, 7), (5, 6), (6, 7), (5, 8), (7, 9)]
        )
        inputs = [1, 0]
        outputs = [9, 8]

        # Heuristic mixture of Pauli and non-Pauli angles ensuring there's no gflow but there's pflow.
        meas_angles: dict[int, float] = {
            **dict.fromkeys(range(4), 0),
            **dict(zip(range(4, 8), (2 * fx_rng.random(4)).tolist())),
        }
        meas_planes = dict.fromkeys(range(8), Plane.XY)
        meas = {i: Measurement(angle, plane) for (i, angle), plane in zip(meas_angles.items(), meas_planes.values())}

        gf, _ = find_gflow(graph=graph, iset=set(inputs), oset=set(outputs), meas_planes=meas_planes)
        pf, _ = find_pauliflow(
            graph=graph, iset=set(inputs), oset=set(outputs), meas_planes=meas_planes, meas_angles=meas_angles
        )

        assert gf is None  # example graph doesn't have gflow
        assert pf is not None  # example graph has Pauli flow

        return OpenGraph(inside=graph, inputs=inputs, outputs=outputs, measurements=meas)

    def test_pattern_generation_determinism_flow(self, fx_rng: Generator) -> None:
        graph: nx.Graph[int] = nx.Graph([(0, 3), (1, 4), (2, 5), (1, 3), (2, 4), (3, 6), (4, 7), (5, 8)])
        inputs = [1, 0, 2]  # non-trivial order to check order is conserved.
        outputs = [7, 6, 8]
        angles = dict(zip(range(6), (2 * fx_rng.random(6)).tolist()))
        meas_planes = dict.fromkeys(range(6), Plane.XY)

        pattern = generate_from_graph(graph, angles, inputs, outputs, meas_planes=meas_planes)
        pattern.standardize()
        pattern.minimize_space()

        repeats = 3  # for testing the determinism of a pattern
        results = [pattern.simulate_pattern(rng=fx_rng) for _ in range(repeats)]

        for i in range(1, 3):
            inner_product = np.dot(results[0].flatten(), results[i].flatten().conjugate())
            assert abs(inner_product) == pytest.approx(1)

        assert pattern.input_nodes == inputs
        assert pattern.output_nodes == outputs

    def test_pattern_generation_determinism_gflow(self, fx_rng: Generator) -> None:
        graph: nx.Graph[int] = nx.Graph([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (3, 6), (1, 6)])
        inputs = [3, 1, 5]
        outputs = [4, 2, 6]
        angles = dict(zip([1, 3, 5], (2 * fx_rng.random(3)).tolist()))
        meas_planes = dict.fromkeys([1, 3, 5], Plane.XY)

        pattern = generate_from_graph(graph, angles, inputs, outputs, meas_planes=meas_planes)
        pattern.standardize()
        pattern.minimize_space()

        repeats = 3  # for testing the determinism of a pattern
        results = [pattern.simulate_pattern(rng=fx_rng) for _ in range(repeats)]

        for i in range(1, 3):
            inner_product = np.dot(results[0].flatten(), results[i].flatten().conjugate())
            assert abs(inner_product) == pytest.approx(1)

        assert pattern.input_nodes == inputs
        assert pattern.output_nodes == outputs

    def test_pattern_generation_determinism_pflow(self, fx_rng: Generator) -> None:
        og = self.get_graph_pflow(fx_rng)
        pattern = og.to_pattern()
        pattern.standardize()
        pattern.minimize_space()

        repeats = 3  # for testing the determinism of a pattern
        results = [pattern.simulate_pattern(rng=fx_rng) for _ in range(repeats)]

        for i in range(1, 3):
            inner_product = np.dot(results[0].flatten(), results[i].flatten().conjugate())
            assert abs(inner_product) == pytest.approx(1)

        assert og.inputs == pattern.input_nodes
        assert og.outputs == pattern.output_nodes

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

    def test_pattern_generation_pflow(self, fx_rng: Generator) -> None:
        og = self.get_graph_pflow(fx_rng)
        pattern = og.to_pattern()

        _, edge_list = pattern.get_graph()
        graph_generated_pattern: nx.Graph[int] = nx.Graph(edge_list)
        assert nx.is_isomorphic(og.inside, graph_generated_pattern)

        pattern.standardize()
        pf_generated_pattern, _ = pauliflow_from_pattern(pattern)
        assert pf_generated_pattern is not None
