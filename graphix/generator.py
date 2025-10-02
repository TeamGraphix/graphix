"""MBQC pattern generator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import graphix.gflow
from graphix.command import E, M, N, X, Z
from graphix.fundamentals import Plane
from graphix.pattern import Pattern

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from collections.abc import Set as AbstractSet

    import networkx as nx

    from graphix.parameter import ExpressionOrFloat


def generate_from_graph(
    graph: nx.Graph[int],
    angles: Mapping[int, ExpressionOrFloat],
    inputs: Iterable[int],
    outputs: Iterable[int],
    meas_planes: Mapping[int, Plane] | None = None,
) -> Pattern:
    r"""Generate the measurement pattern from open graph and measurement angles.

    This function takes an open graph ``G = (nodes, edges, input, outputs)``,
    specified by :class:`networkx.Graph` and two lists specifying input and output nodes.
    Currently we support XY-plane measurements.

    Searches for the flow in the open graph using :func:`graphix.gflow.find_flow` and if found,
    construct the measurement pattern according to the theorem 1 of [NJP 9, 250 (2007)].

    Then, if no flow was found, searches for gflow using :func:`graphix.gflow.find_gflow`,
    from which measurement pattern can be constructed from theorem 2 of [NJP 9, 250 (2007)].

    Then, if no gflow was found, searches for Pauli flow using :func:`graphix.gflow.find_pauliflow`,
    from which measurement pattern can be constructed from theorem 4 of [NJP 9, 250 (2007)].

    The constructed measurement pattern deterministically realize the unitary embedding

    .. math::

        U = \left( \prod_i \langle +_{\alpha_i} |_i \right) E_G N_{I^C},

    where the measurements (bras) with always :math:`\langle+|` bases determined by the measurement
    angles :math:`\alpha_i` are applied to the measuring nodes,
    i.e. the randomness of the measurement is eliminated by the added byproduct commands.

    .. seealso:: :func:`graphix.gflow.find_flow` :func:`graphix.gflow.find_gflow` :func:`graphix.gflow.find_pauliflow` :class:`graphix.pattern.Pattern`

    Parameters
    ----------
    graph : :class:`networkx.Graph`
        Graph on which MBQC should be performed
    angles : dict
        measurement angles for each nodes on the graph (unit of pi), except output nodes
    inputs : list
        list of node indices for input nodes
    outputs : list
        list of node indices for output nodes
    meas_planes : dict
        optional: measurement planes for each nodes on the graph, except output nodes

    Returns
    -------
    pattern : graphix.pattern.Pattern
        constructed pattern.
    """
    inputs_set = set(inputs)
    outputs_set = set(outputs)

    measuring_nodes = set(graph.nodes) - outputs_set

    meas_planes = dict.fromkeys(measuring_nodes, Plane.XY) if not meas_planes else dict(meas_planes)

    # search for flow first
    f, l_k = graphix.gflow.find_flow(graph, inputs_set, outputs_set, meas_planes=meas_planes)
    if f is not None:
        # flow found
        pattern = _flow2pattern(graph, angles, inputs, f, l_k)
        pattern.reorder_output_nodes(outputs)
        return pattern

    # no flow found - we try gflow
    g, l_k = graphix.gflow.find_gflow(graph, inputs_set, outputs_set, meas_planes=meas_planes)
    if g is not None and l_k is not None:
        # gflow found
        pattern = _gflow2pattern(graph, angles, inputs, meas_planes, g, l_k)
        pattern.reorder_output_nodes(outputs)
        return pattern

    # no flow or gflow found - we try pflow
    p, l_k = graphix.gflow.find_pauliflow(graph, inputs_set, outputs_set, meas_planes=meas_planes, meas_angles=angles)
    if p is not None and l_k is not None:
        # pflow found
        pattern = _pflow2pattern(graph, angles, inputs, meas_planes, p, l_k)
        pattern.reorder_output_nodes(outputs)
        return pattern

    raise ValueError("no flow or gflow or pflow found")


def _flow2pattern(
    graph: nx.Graph[int],
    angles: Mapping[int, ExpressionOrFloat],
    inputs: Iterable[int],
    f: Mapping[int, AbstractSet[int]],
    l_k: Mapping[int, int],
) -> Pattern:
    """Construct a measurement pattern from a causal flow according to the theorem 1 of [NJP 9, 250 (2007)]."""
    depth, layers = graphix.gflow.get_layers(l_k)
    pattern = Pattern(input_nodes=inputs)
    for i in set(graph.nodes) - set(inputs):
        pattern.add(N(node=i))
    for e in graph.edges:
        pattern.add(E(nodes=e))
    measured: list[int] = []
    for i in range(depth, 0, -1):  # i from depth, depth-1, ... 1
        for j in layers[i]:
            measured.append(j)
            pattern.add(M(node=j, angle=angles[j]))
            neighbors: set[int] = set()
            for k in f[j]:
                neighbors |= set(graph.neighbors(k))
            for k in neighbors - {j}:
                # if k not in measured:
                pattern.add(Z(node=k, domain={j}))
            (fj,) = f[j]
            pattern.add(X(node=fj, domain={j}))
    return pattern


def _gflow2pattern(
    graph: nx.Graph[int],
    angles: Mapping[int, ExpressionOrFloat],
    inputs: Iterable[int],
    meas_planes: Mapping[int, Plane],
    g: Mapping[int, AbstractSet[int]],
    l_k: Mapping[int, int],
) -> Pattern:
    """Construct a measurement pattern from a generalized flow according to the theorem 2 of [NJP 9, 250 (2007)]."""
    depth, layers = graphix.gflow.get_layers(l_k)
    pattern = Pattern(input_nodes=inputs)
    for i in set(graph.nodes) - set(inputs):
        pattern.add(N(node=i))
    for e in graph.edges:
        pattern.add(E(nodes=e))
    for i in range(depth, 0, -1):  # i from depth, depth-1, ... 1
        for j in layers[i]:
            pattern.add(M(node=j, plane=meas_planes[j], angle=angles[j]))
            odd_neighbors = graphix.gflow.find_odd_neighbor(graph, g[j])
            for k in odd_neighbors - {j}:
                pattern.add(Z(node=k, domain={j}))
            for k in g[j] - {j}:
                pattern.add(X(node=k, domain={j}))
    return pattern


def _pflow2pattern(
    graph: nx.Graph[int],
    angles: Mapping[int, ExpressionOrFloat],
    inputs: Iterable[int],
    meas_planes: Mapping[int, Plane],
    p: Mapping[int, AbstractSet[int]],
    l_k: Mapping[int, int],
) -> Pattern:
    """Construct a measurement pattern from a Pauli flow according to the theorem 4 of [NJP 9, 250 (2007)]."""
    depth, layers = graphix.gflow.get_layers(l_k)
    pattern = Pattern(input_nodes=inputs)
    for i in set(graph.nodes) - set(inputs):
        pattern.add(N(node=i))
    for e in graph.edges:
        pattern.add(E(nodes=e))
    for i in range(depth, 0, -1):  # i from depth, depth-1, ... 1
        for j in layers[i]:
            pattern.add(M(node=j, plane=meas_planes[j], angle=angles[j]))
            odd_neighbors = graphix.gflow.find_odd_neighbor(graph, p[j])
            future_nodes: set[int] = set.union(
                *(nodes for (layer, nodes) in layers.items() if layer < i)
            )  # {k | k > j}, with "j" last corrected node and ">" the Pauli flow ordering
            for k in odd_neighbors & future_nodes:
                pattern.add(Z(node=k, domain={j}))
            for k in p[j] & future_nodes:
                pattern.add(X(node=k, domain={j}))
    return pattern
