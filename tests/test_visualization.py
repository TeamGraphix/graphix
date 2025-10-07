from __future__ import annotations

from math import pi

import matplotlib.pyplot as plt
import pytest

from graphix import gflow, transpiler, visualization


def test_get_pos_from_flow() -> None:
    circuit = transpiler.Circuit(1)
    circuit.h(0)
    pattern = circuit.transpile().pattern
    graph = pattern.extract_graph()
    vin = pattern.input_nodes if pattern.input_nodes is not None else []
    vout = pattern.output_nodes
    meas_planes = pattern.get_meas_plane()
    meas_angles = pattern.get_angles()
    local_clifford = pattern.get_vops()
    vis = visualization.GraphVisualizer(graph, vin, vout, meas_planes, meas_angles, local_clifford)
    f, l_k = gflow.find_flow(graph, set(vin), set(vout), meas_planes)
    assert f is not None
    assert l_k is not None
    pos = vis.get_pos_from_flow(f, l_k)
    assert pos is not None


@pytest.fixture
def mock_plot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)


@pytest.mark.usefixtures("mock_plot")
def test_draw_graph_flow_from_pattern() -> None:
    circuit = transpiler.Circuit(3)
    circuit.cnot(0, 1)
    circuit.cnot(2, 1)
    circuit.rx(0, pi / 3)
    circuit.x(2)
    circuit.cnot(2, 1)
    pattern = circuit.transpile().pattern
    pattern.perform_pauli_measurements(leave_input=True)
    pattern.draw_graph(flow_from_pattern=True, show_measurement_planes=True, node_distance=(0.7, 0.6))
