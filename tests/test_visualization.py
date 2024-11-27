from __future__ import annotations

import networkx as nx

from graphix import gflow, transpiler, visualization


def test_get_pos_from_flow():
    circuit = transpiler.Circuit(1)
    circuit.h(0)
    pattern = circuit.transpile().pattern
    nodes, edges = pattern.get_graph()
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    vin = pattern.input_nodes if pattern.input_nodes is not None else []
    vout = pattern.output_nodes
    meas_planes = pattern.get_meas_plane()
    meas_angles = pattern.get_angles()
    local_clifford = pattern.get_vops()
    vis = visualization.GraphVisualizer(g, vin, vout, meas_planes, meas_angles, local_clifford)
    f, l_k = gflow.find_flow(g, set(vin), set(vout), meas_planes)
    pos = vis.get_pos_from_flow(f, l_k)
    assert pos is not None
