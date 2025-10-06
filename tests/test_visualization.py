from __future__ import annotations

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
