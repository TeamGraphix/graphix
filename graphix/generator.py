"""
MBQC pattern generator

"""


import numpy as np
from graphix.pattern import Pattern
from graphix.gflow import flow, gflow, get_layers, find_odd_neighbor


def generate_from_graph(graph, angles, inputs, outputs, timeout=100):
    r"""Generate the measurement pattern from open graph and measurement angles.

    This function takes an open graph G = (nodes, edges, input, outputs),
    specified by networks.Graph and two lists specifying input and output nodes.
    Currently we support XY-plane measurements.

    Searches for the flow in the open graph using :func:`flow` and if found,
    construct the measurement pattern according to the theorem 1 of [NJP 9, 250 (2007)].

    Then, if no flow was found, searches for gflow using :func:`gflow`,
    from which measurement pattern can be constructed from theorem 2 of [NJP 9, 250 (2007)].

    The constructed measurement pattern deterministically realize the unitary embedding

    .. math::

        U = \left( \prod_i \langle +_{\alpha_i} |_i \right) E_G N_{I^C},

    where the measurements (bras) with always :math:`\langle+|` bases determined by the measurement
    angles :math:`\alpha_i` are applied to the measuring nodes,
    i.e. the randomness of the measurement is eliminated by the added byproduct commands.

    .. seealso:: :func:`flow` :func:`gflow` :class:`graphix.pattern.Pattern`

    Parameters
    ----------
    graph : networkx.Graph
        graph on which MBQC should be performed
    angles : dict
        measurement angles for each nodes on the graph (unit of pi), except output nodes
    inputs : list
        list of node indices for input nodes
    outputs : list
        list of node indices for output nodes
    timeout : int
        optional argument for flow and gflow search depth

    Returns
    -------
    pattern : graphix.pattern.Pattern object
        constructed pattern.
    """
    assert len(inputs) == len(outputs)
    measuring_nodes = list(set(graph.nodes) - set(outputs) - set(inputs))

    # search for flow first
    f, l_k = flow(graph, set(inputs), set(outputs), timeout=timeout)
    if f:
        # flow found
        depth, layers = get_layers(l_k)
        pattern = Pattern(len(inputs))
        pattern.seq = [["N", i] for i in inputs]
        for i in set(graph.nodes) - set(inputs):
            pattern.seq.append(["N", i])
        for e in graph.edges:
            pattern.seq.append(["E", e])
        measured = []
        for i in range(depth, 0, -1):  # i from depth, depth-1, ... 1
            for j in layers[i]:
                measured.append(j)
                pattern.seq.append(["M", j, "XY", angles[j], [], []])
                for k in set(graph.neighbors(f[j])) - set([j]):
                    if k not in measured:
                        pattern.seq.append(["Z", k, [j]])
                pattern.seq.append(["X", f[j], [j]])
        pattern.output_nodes = outputs
        pattern.Nnode = len(graph.nodes)
    else:
        # no flow found - we try gflow
        g, l_k = gflow(graph, set(inputs), set(outputs), timeout=timeout)
        if g:
            # gflow found
            depth, layers = get_layers(l_k)
            pattern = Pattern(len(inputs))
            pattern.seq = [["N", i] for i in inputs]
            for i in set(graph.nodes) - set(inputs):
                pattern.seq.append(["N", i])
            for e in graph.edges:
                pattern.seq.append(["E", e])
            remaining = set(measuring_nodes)
            for i in range(depth, 0, -1):  # i from depth, depth-1, ... 1
                for j in layers[i]:
                    pattern.seq.append(["M", j, "XY", angles[j], [], []])
                    remaining = remaining - set([j])
                    odd_neighbors = find_odd_neighbor(graph, remaining, set(g[j]))
                    for k in odd_neighbors:
                        pattern.seq.append(["Z", k, [j]])
                    for k in set(g[j]) - set([j]):
                        pattern.seq.append(["X", k, [j]])
            pattern.output_nodes = outputs
            pattern.Nnode = len(graph.nodes)
        else:
            raise ValueError("no flow or gflow found")

    return pattern
