"""Flow finding algorithm.

For a given underlying graph (G, I, O, meas_plane), this method finds a (generalized) flow [NJP 9, 250 (2007)] in polynomial time.
In particular, this outputs gflow with minimum depth, maximally delayed gflow.

Ref: Mhalla and Perdrix, International Colloquium on Automata,
Languages, and Programming (Springer, 2008), pp. 857-868.
Ref: Backens et al., Quantum 5, 421 (2021).

"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import networkx as nx
from typing_extensions import assert_never

import graphix.opengraph
from graphix.command import CommandKind
from graphix.find_pflow import find_pflow as _find_pflow
from graphix.fundamentals import Axis, Plane
from graphix.measurements import Measurement, PauliMeasurement
from graphix.parameter import Placeholder

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet

    from graphix.parameter import ExpressionOrFloat
    from graphix.pattern import Pattern


# TODO: This should be ensured by type-checking.
def check_meas_planes(meas_planes: dict[int, Plane]) -> None:
    """Check that all planes are valid planes."""
    for node, plane in meas_planes.items():
        if not isinstance(plane, Plane):
            raise TypeError(f"Measure plane for {node} is `{plane}`, which is not an instance of `Plane`")


# NOTE: In a future version this function will take an `OpenGraph` object as input.
def find_gflow(
    graph: nx.Graph[int],
    iset: AbstractSet[int],
    oset: AbstractSet[int],
    meas_planes: Mapping[int, Plane],
    mode: str = "single",  # noqa: ARG001 Compatibility with old API
) -> tuple[dict[int, set[int]], dict[int, int]] | tuple[None, None]:
    r"""Return a maximally delayed general flow (gflow) of the input open graph if it exists.

    Parameters
    ----------
    graph: :class:`networkx.Graph`
        Graph (including input and output).
    iset: AbstractSet[int]
        Set of input nodes.
    oset: AbstractSet[int]
        Set of output nodes.
    meas_planes: Mapping[int, Plane]
        Measurement planes for each qubit. meas_planes[i] is the measurement plane for qubit i.
    mode: str
        Deprecated. Reminiscent of old API, it will be removed in future versions.

    Returns
    -------
    dict[int, set[int]]
        Gflow correction function. In a given pair (key, value), value is the set of qubits to be corrected for the measurement of qubit key.
    dict[int, int]
        Partial order between corrected qubits, such that the pair (key, value) corresponds to (node, depth).

    or None, None
        if the input open graph does not have gflow.

    Notes
    -----
    This function implements the algorithm in [1], see module graphix.find_pflow.
    See [1] or [2] for a definition of gflow.

    References
    ----------
    [1] Mitosek and Backens, 2024 (arXiv:2410.23439).
    [2] Backens et al., Quantum 5, 421 (2021).
    """
    meas = {node: Measurement(Placeholder("Angle"), plane) for node, plane in meas_planes.items()}
    og = graphix.opengraph.OpenGraph(
        inside=graph,
        inputs=list(iset),
        outputs=list(oset),
        measurements=meas,
    )
    gf = _find_pflow(og)
    if gf is None:
        return None, None  # This is to comply with old API. It will be change in the future to `None``
    return gf[0], gf[1]


def find_flow(
    graph: nx.Graph[int],
    iset: set[int],
    oset: set[int],
    meas_planes: dict[int, Plane] | None = None,
) -> tuple[dict[int, set[int]], dict[int, int]]:
    """Causal flow finding algorithm.

    For open graph g with input, output, and measurement planes, this returns causal flow.
    For more detail of causal flow, see Danos and Kashefi, PRA 74, 052310 (2006).

    Original algorithm by Mhalla and Perdrix,
    International Colloquium on Automata, Languages, and Programming (2008),
    pp. 857-868.

    Parameters
    ----------
    graph: :class:`networkx.Graph`
        Graph (incl. input and output)
    iset: set
        set of node labels for input
    oset: set
        set of node labels for output
    meas_planes: dict(int, Plane)
        measurement planes for each qubits. meas_planes[i] is the measurement plane for qubit i.
        Note that an underlying graph has a causal flow only if all measurement planes are Plane.XY.
        If not specified, all measurement planes are interpreted as Plane.XY.

    Returns
    -------
    f: list of nodes
        causal flow function. f[i] is the qubit to be measured after qubit i.
    l_k: dict
        layers obtained by gflow algorithm. l_k[d] is a node set of depth d.
    """
    check_meas_planes(meas_planes)
    nodes = set(graph.nodes)
    edges = set(graph.edges)

    if meas_planes is None:
        meas_planes = dict.fromkeys(nodes - oset, Plane.XY)

    for plane in meas_planes.values():
        if plane != Plane.XY:
            return None, None

    l_k = dict.fromkeys(nodes, 0)
    f = {}
    k = 1
    v_c = oset - iset
    return flowaux(nodes, edges, iset, oset, v_c, f, l_k, k)


def flowaux(
    nodes: set[int],
    edges: set[tuple[int, int]],
    iset: set[int],
    oset: set[int],
    v_c: set[int],
    f: dict[int, set[int]],
    l_k: dict[int, int],
    k: int,
):
    """Find one layer of the flow.

    Ref: Mhalla and Perdrix, International Colloquium on Automata,
    Languages, and Programming (Springer, 2008), pp. 857-868.

    Parameters
    ----------
    nodes: set
        labels of all qubits (nodes)
    edges: set
        edges
    iset: set
        set of node labels for input
    oset: set
        set of node labels for output
    v_c: set
        correction candidate qubits
    f: dict
        flow function. f[i] is the qubit to be measured after qubit i.
    l_k: dict
        layers obtained by flow algorithm. l_k[d] is a node set of depth d.
    k: int
        current layer number.
    meas_planes: dict
        measurement planes for each qubits. meas_planes[i] is the measurement plane for qubit i.

    Outputs
    -------
    f: list of nodes
        causal flow function. f[i] is the qubit to be measured after qubit i.
    l_k: dict
        layers obtained by gflow algorithm. l_k[d] is a node set of depth d.
    """
    v_out_prime = set()
    c_prime = set()

    for q in v_c:
        nb = search_neighbor(q, edges)
        p_set = nb & (nodes - oset)
        if len(p_set) == 1:
            # Iterate over p_set assuming there is only one element p
            (p,) = p_set
            f[p] = {q}
            l_k[p] = k
            v_out_prime |= {p}
            c_prime |= {q}
    # determine whether there exists flow
    if not v_out_prime:
        if oset == nodes:
            return f, l_k
        return None, None
    return flowaux(
        nodes,
        edges,
        iset,
        oset | v_out_prime,
        (v_c - c_prime) | (v_out_prime & (nodes - iset)),
        f,
        l_k,
        k + 1,
    )


# NOTE: In a future version this function will take an `OpenGraph` object as input.
def find_pauliflow(
    graph: nx.Graph[int],
    iset: AbstractSet[int],
    oset: AbstractSet[int],
    meas_planes: Mapping[int, Plane],
    meas_angles: Mapping[int, ExpressionOrFloat],
    mode: str = "single",  # noqa: ARG001 Compatibility with old API
) -> tuple[dict[int, set[int]], dict[int, int]] | tuple[None, None]:
    r"""Return a maximally delayed Pauli flow of the input open graph if it exists.

    Parameters
    ----------
    graph: :class:`networkx.Graph`
        Graph (including input and output).
    iset: AbstractSet[int]
        Set of input nodes.
    oset: AbstractSet[int]
        Set of output nodes.
    meas_planes: Mapping[int, Plane]
        Measurement planes for each qubit. meas_planes[i] is the measurement plane for qubit i.
    meas_angles: Mapping[int, ExpressionOrFloat]
        Measurement angles for each qubit. meas_angles[i] is the measurement angle for qubit i.
    mode: str
        Deprecated. Reminiscent of old API, it will be removed in future versions.

    Returns
    -------
    dict[int, set[int]]
        Pauli flow correction function. In a given pair (key, value), value is the set of qubits to be corrected for the measurement of qubit key.
    dict[int, int]
        Partial order between corrected qubits, such that the pair (key, value) corresponds to (node, depth).

    or None, None
        if the input open graph does not have gflow.

    Notes
    -----
    This function implements the algorithm in [1], see module graphix.find_pflow.
    See [1] or [2] for a definition of Pauli flow.

    References
    ----------
    [1] Mitosek and Backens, 2024 (arXiv:2410.23439).
    [2] Browne et al., 2007 New J. Phys. 9 250 (arXiv:quant-ph/0702212)
    """
    meas = {node: Measurement(angle, meas_planes[node]) for node, angle in meas_angles.items()}
    og = graphix.opengraph.OpenGraph(
        inside=graph,
        inputs=list(iset),
        outputs=list(oset),
        measurements=meas,
    )
    pf = _find_pflow(og)
    if pf is None:
        return None, None  # This is to comply with old API. It will be change in the future to `None``
    return pf[0], pf[1]


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
    if not pattern.is_standard(strict=True):
        raise ValueError("The pattern should be standardized first.")
    meas_planes = pattern.get_meas_plane()
    for plane in meas_planes.values():
        if plane != Plane.XY:
            return None, None
    g = nx.Graph()
    nodes, edges = pattern.get_graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    input_nodes = pattern.input_nodes if not pattern.input_nodes else set()
    output_nodes = set(pattern.output_nodes)
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

    if verify_flow(g, input_nodes, output_nodes, xflow):  # if xflow is valid
        zflow_from_xflow = {}
        for node, corrections in deepcopy(xflow).items():
            cand = find_odd_neighbor(g, corrections) - {node}
            if cand:
                zflow_from_xflow[node] = cand
        if zflow_from_xflow != zflow:  # if zflow is consistent with xflow
            return None, None
        return xflow, l_k
    return None, None


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
    if not pattern.is_standard(strict=True):
        raise ValueError("The pattern should be standardized first.")
    g = nx.Graph()
    nodes, edges = pattern.get_graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    input_nodes = set(pattern.input_nodes) if pattern.input_nodes else set()
    output_nodes = set(pattern.output_nodes)
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

    if verify_gflow(g, input_nodes, output_nodes, xflow, meas_planes):  # if xflow is valid
        zflow_from_xflow = {}
        for node, corrections in deepcopy(xflow).items():
            cand = find_odd_neighbor(g, corrections) - {node}
            if cand:
                zflow_from_xflow[node] = cand
        if zflow_from_xflow != zflow:  # if zflow is consistent with xflow
            return None, None
        return xflow, l_k
    return None, None


# TODO: Shouldn't call `find_pauliflow`
def pauliflow_from_pattern(
    pattern: Pattern,
    mode="single",  # noqa: ARG001 Compatibility with old API
) -> tuple[dict[int, set[int]], dict[int, int]] | tuple[None, None]:
    """Check if the pattern has a valid Pauliflow. If so, return the Pauliflow and layers.

    Parameters
    ----------
    pattern: Pattern
        pattern to be based on
    mode: str
        The Pauliflow finding algorithm can yield multiple equivalent solutions. So there are two options
            - "single": Returns a single solution
            - "all": Returns all possible solutions

        Optional. Default is "single".

    Returns
    -------
    p: dict
        Pauli flow function. p[i] is the set of qubits to be corrected for the measurement of qubit i.
    l_k: dict
        layers obtained by Pauli flow algorithm. l_k[d] is a node set of depth d.
    """
    if not pattern.is_standard(strict=True):
        raise ValueError("The pattern should be standardized first.")
    g: nx.Graph[int] = nx.Graph()
    nodes, edges = pattern.get_graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    input_nodes = set(pattern.input_nodes) if pattern.input_nodes else set()
    output_nodes = set(pattern.output_nodes) if pattern.output_nodes else set()
    meas_planes = pattern.get_meas_plane()
    meas_angles = pattern.get_angles()

    return find_pauliflow(g, input_nodes, output_nodes, meas_planes, meas_angles)


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
    nodes, _ = pattern.get_graph()
    nodes = set(nodes)
    xflow = {}
    zflow = {}
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


def search_neighbor(node: int, edges: set[tuple[int, int]]) -> set[int]:
    """Find neighborhood of node in edges. This is an ancillary method for `flowaux()`.

    Parameter
    -------
    node: int
        target node number whose neighboring nodes will be collected
    edges: set of taples
        set of edges in the graph

    Outputs
    ------
    N: list of ints
        neighboring nodes
    """
    nb = set()
    for edge in edges:
        if node == edge[0]:
            nb |= {edge[1]}
        elif node == edge[1]:
            nb |= {edge[0]}
    return nb


def get_min_depth(l_k: Mapping[int, int]) -> int:
    """Get minimum depth of graph.

    Parameters
    ----------
    l_k: dict
        layers obtained by flow or gflow

    Returns
    -------
    d: int
        minimum depth of graph
    """
    return max(l_k.values())


def find_odd_neighbor(graph: nx.Graph[int], vertices: AbstractSet[int]) -> set[int]:
    """Return the set containing the odd neighbor of a set of vertices.

    Parameters
    ----------
    graph : :class:`networkx.Graph`
        Underlying graph
    vertices : set
        set of nodes indices to find odd neighbors

    Returns
    -------
    odd_neighbors : set
        set of indices for odd neighbor of set `vertices`.
    """
    odd_neighbors = set()
    for vertex in vertices:
        neighbors = set(graph.neighbors(vertex))
        odd_neighbors ^= neighbors
    return odd_neighbors


def get_layers(l_k: Mapping[int, int]) -> tuple[int, dict[int, set[int]]]:
    """Get components of each layer.

    Parameters
    ----------
    l_k: dict
        layers obtained by flow or gflow algorithms

    Returns
    -------
    d: int
        minimum depth of graph
    layers: dict of set
        components of each layer
    """
    d = get_min_depth(l_k)
    layers: dict[int, set[int]] = {k: set() for k in range(d + 1)}
    for i, val in l_k.items():
        layers[val] |= {i}
    return d, layers


def get_dependence_flow(
    inputs: set[int],
    flow: dict[int, set[int]],
    odd_flow: dict[int, set[int]],
) -> dict[int, set[int]]:
    """Get dependence flow from flow.

    Parameters
    ----------
    inputs: set[int]
        set of input nodes
    flow: dict[int, set]
        flow function. flow[i] is the set of qubits to be corrected for the measurement of qubit i.
    odd_flow: dict[int, set]
        odd neighbors of flow or gflow.
        odd_flow[i] is the set of odd neighbors of f(i), Odd(f(i)).

    Returns
    -------
    dependence_flow: dict[int, set]
        dependence flow function. dependence_flow[i] is the set of qubits to be corrected for the measurement of qubit i.
    """
    dependence_flow = {u: set() for u in inputs}
    # concatenate flow and odd_flow
    combined_flow = {}
    for node, corrections in flow.items():
        combined_flow[node] = corrections | odd_flow[node]
    for node, corrections in combined_flow.items():
        for correction in corrections:
            if correction not in dependence_flow:
                dependence_flow[correction] = set()
            dependence_flow[correction] |= {node}
    return dependence_flow


def get_dependence_pauliflow(
    inputs: set[int],
    flow: dict[int, set[int]],
    odd_flow: dict[int, set[int]],
    ls: tuple[set[int], set[int], set[int]],
):
    """Get dependence flow from Pauli flow.

    Parameters
    ----------
    inputs: set[int]
        set of input nodes
    flow: dict[int, set[int]]
        Pauli flow function. p[i] is the set of qubits to be corrected for the measurement of qubit i.
    odd_flow: dict[int, set[int]]
        odd neighbors of Pauli flow or gflow. Odd(p(i))
    ls: tuple
        ls = (l_x, l_y, l_z) where l_x, l_y, l_z are sets of qubits whose measurement operators are X, Y, Z, respectively.

    Returns
    -------
    dependence_pauliflow: dict[int, set[int]]
        dependence flow function. dependence_pauliflow[i] is the set of qubits to be corrected for the measurement of qubit i.
    """
    l_x, l_y, l_z = ls
    dependence_pauliflow = {u: set() for u in inputs}
    # concatenate p and odd_p
    combined_flow = {}
    for node, corrections in flow.items():
        combined_flow[node] = (corrections - (l_x | l_y)) | (odd_flow[node] - (l_y | l_z))
        for ynode in l_y:
            if ynode in corrections.symmetric_difference(odd_flow[node]):
                combined_flow[node] |= {ynode}
    for node, corrections in combined_flow.items():
        for correction in corrections:
            if correction not in dependence_pauliflow:
                dependence_pauliflow[correction] = set()
            dependence_pauliflow[correction] |= {node}
    return dependence_pauliflow


def get_layers_from_flow(
    flow: dict[int, set],
    odd_flow: dict[int, set],
    inputs: set[int],
    outputs: set[int],
    ls: tuple[set[int], set[int], set[int]] | None = None,
) -> tuple[dict[int, set], int]:
    """Get layers from flow (incl. gflow, Pauli flow).

    Parameters
    ----------
    flow: dict[int, set]
        flow function. flow[i] is the set of qubits to be corrected for the measurement of qubit i.
    odd_flow: dict[int, set]
        odd neighbors of flow or gflow. Odd(f(node))
    inputs: set
        set of input nodes
    outputs: set
        set of output nodes
    ls: tuple
        ls = (l_x, l_y, l_z) where l_x, l_y, l_z are sets of qubits whose measurement operators are X, Y, Z, respectively.
        If not None, the layers are obtained based on Pauli flow.

    Returns
    -------
    layers: dict[int, set]
        layers obtained from flow
    depth: int
        depth of the layers

    Raises
    ------
    ValueError
        If the flow is not valid(e.g. there is no partial order).
    """
    layers = {}
    depth = 0
    if ls is None:
        dependence_flow = get_dependence_flow(inputs, odd_flow, flow)
    else:
        dependence_flow = get_dependence_pauliflow(inputs, flow, odd_flow, ls)
    left_nodes = set(flow.keys())
    for output in outputs:
        if output in left_nodes:
            raise ValueError("Invalid flow")
    while True:
        layers[depth] = set()
        for node in left_nodes:
            if node not in dependence_flow or len(dependence_flow[node]) == 0 or dependence_flow[node] == {node}:
                layers[depth] |= {node}
        left_nodes -= layers[depth]
        for node in left_nodes:
            dependence_flow[node] -= layers[depth]
        if len(layers[depth]) == 0:
            if len(left_nodes) == 0:
                layers[depth] = outputs
                depth += 1
                break
            raise ValueError("Invalid flow")
        depth += 1
    return layers, depth


def verify_flow(
    graph: nx.Graph,
    iset: set[int],
    oset: set[int],
    flow: dict[int, set],
    meas_planes: dict[int, Plane] | None = None,
) -> bool:
    """Check whether the flow is valid.

    Parameters
    ----------
    graph: :class:`networkx.Graph`
        Graph (incl. input and output)
    flow: dict[int, set]
        flow function. flow[i] is the set of qubits to be corrected for the measurement of qubit i.
    meas_planes: dict[int, str]
        optional: measurement planes for each qubits. meas_planes[i] is the measurement plane for qubit i.


    Returns
    -------
    valid_flow: bool
        True if the flow is valid. False otherwise.
    """
    if meas_planes is None:
        meas_planes = {}
    check_meas_planes(meas_planes)
    valid_flow = True
    non_outputs = set(graph.nodes) - oset
    # if meas_planes is given, check whether all measurement planes are "XY"
    for node, plane in meas_planes.items():
        if plane != Plane.XY or node not in non_outputs:
            return False

    odd_flow = {node: find_odd_neighbor(graph, corrections) for node, corrections in flow.items()}

    try:
        _, _ = get_layers_from_flow(flow, odd_flow, iset, oset)
    except ValueError:
        return False
    # check if v ~ f(v) for each node
    edges = set(graph.edges)
    for node, corrections in flow.items():
        if len(corrections) > 1:
            return False
        correction = next(iter(corrections))
        if (node, correction) not in edges and (correction, node) not in edges:
            return False
    return valid_flow


def verify_gflow(
    graph: nx.Graph,
    iset: set[int],
    oset: set[int],
    gflow: dict[int, set],
    meas_planes: dict[int, Plane],
) -> bool:
    """Check whether the gflow is valid.

    Parameters
    ----------
    graph: :class:`networkx.Graph`
        Graph (incl. input and output)
    iset: set
        set of node labels for input
    oset: set
        set of node labels for output
    gflow: dict[int, set]
        gflow function. gflow[i] is the set of qubits to be corrected for the measurement of qubit i.
        .. seealso:: :func:`find_gflow`
    meas_planes: dict[int, str]
        measurement planes for each qubits. meas_planes[i] is the measurement plane for qubit i.

    Returns
    -------
    valid_gflow: bool
        True if the gflow is valid. False otherwise.
    """
    check_meas_planes(meas_planes)
    valid_gflow = True
    non_outputs = set(graph.nodes) - oset
    odd_flow = {}
    for non_output in non_outputs:
        if non_output not in gflow:
            gflow[non_output] = set()
            odd_flow[non_output] = set()
        else:
            odd_flow[non_output] = find_odd_neighbor(graph, gflow[non_output])

    try:
        _, _ = get_layers_from_flow(gflow, odd_flow, iset, oset)
    except ValueError:
        return False

    # check for each measurement plane
    for node, plane in meas_planes.items():
        # index = node_order.index(node)
        if plane == Plane.XY:
            valid_gflow &= (node not in gflow[node]) and (node in odd_flow[node])
        elif plane == Plane.XZ:
            valid_gflow &= (node in gflow[node]) and (node in odd_flow[node])
        elif plane == Plane.YZ:
            valid_gflow &= (node in gflow[node]) and (node not in odd_flow[node])

    return valid_gflow


def verify_pauliflow(
    graph: nx.Graph,
    iset: set[int],
    oset: set[int],
    pauliflow: dict[int, set[int]],
    meas_planes: dict[int, Plane],
    meas_angles: dict[int, float],
) -> bool:
    """Check whether the Pauliflow is valid.

    Parameters
    ----------
    graph: :class:`networkx.Graph`
        Graph (incl. input and output)
    iset: set
        set of node labels for input
    oset: set
        set of node labels for output
    pauliflow: dict[int, set]
        Pauli flow function. pauliflow[i] is the set of qubits to be corrected for the measurement of qubit i.
    meas_planes: dict[int, Plane]
        measurement planes for each qubits. meas_planes[i] is the measurement plane for qubit i.
    meas_angles: dict[int, float]
        measurement angles for each qubits. meas_angles[i] is the measurement angle for qubit i.

    Returns
    -------
    valid_pauliflow: bool
        True if the Pauliflow is valid. False otherwise.
    """
    check_meas_planes(meas_planes)
    l_x, l_y, l_z = get_pauli_nodes(meas_planes, meas_angles)

    valid_pauliflow = True
    non_outputs = set(graph.nodes) - oset
    odd_flow = {}
    for non_output in non_outputs:
        if non_output not in pauliflow:
            pauliflow[non_output] = set()
            odd_flow[non_output] = set()
        else:
            odd_flow[non_output] = find_odd_neighbor(graph, pauliflow[non_output])

    try:
        layers, depth = get_layers_from_flow(pauliflow, odd_flow, iset, oset, (l_x, l_y, l_z))
    except ValueError:
        return False
    node_order = []
    for d in range(depth):
        node_order.extend(list(layers[d]))

    for node, plane in meas_planes.items():
        if node in l_x:
            valid_pauliflow &= node in odd_flow[node]
        elif node in l_z:
            valid_pauliflow &= node in pauliflow[node]
        elif node in l_y:
            valid_pauliflow &= node in pauliflow[node].symmetric_difference(odd_flow[node])
        elif plane == Plane.XY:
            valid_pauliflow &= (node not in pauliflow[node]) and (node in odd_flow[node])
        elif plane == Plane.XZ:
            valid_pauliflow &= (node in pauliflow[node]) and (node in odd_flow[node])
        elif plane == Plane.YZ:
            valid_pauliflow &= (node in pauliflow[node]) and (node not in odd_flow[node])

    return valid_pauliflow


def get_input_from_flow(flow: dict[int, set]) -> set:
    """Get input nodes from flow.

    Parameters
    ----------
    flow: dict[int, set]
        flow function. flow[i] is the set of qubits to be corrected for the measurement of qubit i.

    Returns
    -------
    inputs: set
        set of input nodes
    """
    non_output = set(flow.keys())
    for correction in flow.values():
        non_output -= correction
    return non_output


def get_output_from_flow(flow: dict[int, set]) -> set:
    """Get output nodes from flow.

    Parameters
    ----------
    flow: dict[int, set]
        flow function. flow[i] is the set of qubits to be corrected for the measurement of qubit i.

    Returns
    -------
    outputs: set
        set of output nodes
    """
    non_outputs = set(flow.keys())
    non_inputs = set()
    for correction in flow.values():
        non_inputs |= correction
    return non_inputs - non_outputs


def get_pauli_nodes(
    meas_planes: dict[int, Plane], meas_angles: Mapping[int, ExpressionOrFloat]
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
    check_meas_planes(meas_planes)
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
            assert_never(pm.axis)
    return l_x, l_y, l_z
