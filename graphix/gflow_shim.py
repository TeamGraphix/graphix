"""Shim for old gflow.py."""

from __future__ import annotations

import copy

import networkx as nx
import typing_extensions

from graphix import gflow
from graphix.command import CommandKind
from graphix.fundamentals import Axis, Plane
from graphix.measurements import PauliMeasurement

if typing_extensions.TYPE_CHECKING:
    from graphix.pattern import Pattern


def get_pauli_nodes(
    meas_planes: dict[int, Plane], meas_angles: dict[int, float]
) -> tuple[set[int], set[int], set[int]]:
    """Get sets of nodes measured in X, Y, Z basis.

    Parameters
    ----------
    meas_planes: dict[int, Plane]
        measurement planes for each node.
    meas_angles: dict[int, float]
        measurement angles for each node.

    Returns
    -------
    l_x: set
        set of nodes measured in X basis.
    l_y: set
        set of nodes measured in Y basis.
    l_z: set
        set of nodes measured in Z basis.
    """
    l_x, l_y, l_z = set(), set(), set()
    for node, plane in meas_planes.items():
        pm = PauliMeasurement.try_from(plane, meas_angles[node])
        if pm is None:
            continue
        if pm.axis == Axis.X:
            l_x |= {node}
        elif pm.axis == Axis.Y:
            l_y |= {node}
        elif pm.axis == Axis.Z:
            l_z |= {node}
        else:
            typing_extensions.assert_never(pm.axis)
    return l_x, l_y, l_z


def flow_from_pattern(pattern: Pattern) -> tuple[dict[int, set[int]], dict[int, int]]:
    """Check if the pattern has a valid flow. If so, return the flow and layers.

    Parameters
    ----------
    pattern: Pattern
        pattern to be based on

    Returns
    -------
    f: dict
        flow function. g[i] is the set of qubits to be corrected for the measurement of qubit i.
    l_k: dict
        layers obtained by flow algorithm. l_k[d] is a node set of depth d.
    """
    meas_planes = pattern.get_meas_plane()
    for plane in meas_planes.values():
        if plane != Plane.XY:
            return None, None
    g = nx.Graph()
    nodes, edges = pattern.get_graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    nodes = set(nodes)

    layers = pattern.get_layers()
    l_k = {}
    for l in layers[1]:
        for n in layers[1][l]:
            l_k[n] = l
    lmax = max(l_k.values()) if l_k else 0
    for node, val in l_k.items():
        l_k[node] = lmax - val + 1
    for output_node in pattern.output_nodes:
        l_k[output_node] = 0

    xflow, zflow = get_corrections_from_pattern(pattern)

    # Verification skipped
    zflow_from_xflow = {}
    for node, corrections in copy.deepcopy(xflow).items():
        cand = gflow.odd_neighbor(g, corrections) - {node}
        if cand:
            zflow_from_xflow[node] = cand
    if zflow_from_xflow != zflow:  # if zflow is consistent with xflow
        return None, None
    return xflow, l_k


def gflow_from_pattern(pattern: Pattern) -> tuple[dict[int, set[int]], dict[int, int]]:
    """Check if the pattern has a valid gflow. If so, return the gflow and layers.

    Parameters
    ----------
    pattern: Pattern
        pattern to be based on

    Returns
    -------
    g: dict
        gflow function. g[i] is the set of qubits to be corrected for the measurement of qubit i.
    l_k: dict
        layers obtained by gflow algorithm. l_k[d] is a node set of depth d.
    """
    g = nx.Graph()
    nodes, edges = pattern.get_graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    meas_planes = pattern.get_meas_plane()
    nodes = set(nodes)

    layers = pattern.get_layers()
    l_k = {}
    for l in layers[1]:
        for n in layers[1][l]:
            l_k[n] = l
    lmax = max(l_k.values()) if l_k else 0
    for node, val in l_k.items():
        l_k[node] = lmax - val + 1
    for output_node in pattern.output_nodes:
        l_k[output_node] = 0

    xflow, zflow = get_corrections_from_pattern(pattern)
    for node, plane in meas_planes.items():
        if plane in {Plane.XZ, Plane.YZ}:
            if node not in xflow:
                xflow[node] = {node}
            xflow[node] |= {node}

    # Verification skipped
    zflow_from_xflow = {}
    for node, corrections in copy.deepcopy(xflow).items():
        cand = gflow.odd_neighbor(g, corrections) - {node}
        if cand:
            zflow_from_xflow[node] = cand
    if zflow_from_xflow != zflow:  # if zflow is consistent with xflow
        return None, None
    return xflow, l_k


def get_corrections_from_pattern(pattern: Pattern) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """Get x and z corrections from pattern.

    Parameters
    ----------
    pattern: graphix.Pattern object
        pattern to be based on

    Returns
    -------
    xflow: dict
        xflow function. xflow[i] is the set of qubits to be corrected in the X basis for the measurement of qubit i.
    zflow: dict
        zflow function. zflow[i] is the set of qubits to be corrected in the Z basis for the measurement of qubit i.
    """
    nodes_, _ = pattern.get_graph()
    nodes = set(nodes_)
    xflow: dict[int, set[int]] = {}
    zflow: dict[int, set[int]] = {}
    for cmd in pattern:
        if cmd.kind == CommandKind.M:
            target = cmd.node
            xflow_source = cmd.s_domain & nodes
            zflow_source = cmd.t_domain & nodes
            for node in xflow_source:
                if node not in xflow:
                    xflow[node] = set()
                xflow[node] |= {target}
            for node in zflow_source:
                if node not in zflow:
                    zflow[node] = set()
                zflow[node] |= {target}
        if cmd.kind == CommandKind.X:
            target = cmd.node
            xflow_source = cmd.domain & nodes
            for node in xflow_source:
                if node not in xflow:
                    xflow[node] = set()
                xflow[node] |= {target}
        if cmd.kind == CommandKind.Z:
            target = cmd.node
            zflow_source = cmd.domain & nodes
            for node in zflow_source:
                if node not in zflow:
                    zflow[node] = set()
                zflow[node] |= {target}
    return xflow, zflow
