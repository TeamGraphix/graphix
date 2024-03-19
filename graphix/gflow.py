"""flow finding algorithm

For a given underlying graph (G, I, O, meas_plane), this method finds a (generalized) flow [NJP 9, 250 (2007)]
in polynomincal time.
In particular, this outputs gflow with minimum depth, maximally delayed gflow.

Ref: Mhalla and Perdrix, International Colloquium on Automata,
Languages, and Programming (Springer, 2008), pp. 857-868.
Ref: Backens et al., Quantum 5, 421 (2021).

"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphix.pattern import Pattern

from itertools import product

from copy import deepcopy
import networkx as nx
import numpy as np
import sympy as sp

from graphix.linalg import MatGF2


def find_gflow(
    graph: nx.Graph,
    input: set[int],
    output: set[int],
    meas_planes: dict[int, str],
    mode: str = "single",
) -> tuple[dict[int, set[int]], dict[int, int]]:
    """Maximally delayed gflow finding algorithm

    For open graph g with input, output, and measurement planes, this returns maximally delayed gflow.

    gflow consist of function g(i) where i is the qubit labels,
    and strict partial ordering < or layers labels l_k where each element
    specify the order of qubits to be measured to maintain determinism in MBQC.
    In practice, we must measure qubits in order specified in array l_k (increasing order
    of l_k from 1), and for each measurements of qubit i we must perform corrections on
    qubits in g(i), depending on the measurement outcome.

    For more details of gflow, see Browne et al., NJP 9, 250 (2007).
    We use the extended gflow finding algorithm in Backens et al., Quantum 5, 421 (2021).

    Parameters
    ----------
    graph: nx.Graph
        graph (incl. in and out)
    input: set
        set of node labels for input
    output: set
        set of node labels for output
    meas_planes: dict
        measurement planes for each qubits. meas_planes[i] is the measurement plane for qubit i.
    mode: str(optional)
        The gflow finding algorithm can yield multiple equivalent solutions. So there are three options
            - "single": Returrns a single solution
            - "all": Returns all possible solutions
            - "abstract": Returns an abstract solution. Uncertainty is represented with sympy.Symbol objects,
              requiring user substitution to get a concrete answer.

        Default is "single".
    pattern: graphix.Pattern object(optional)
        pattern to be based on. This is used only when mode is "pattern".

    Returns
    -------
    g: dict
        gflow function. g[i] is the set of qubits to be corrected for the measurement of qubit i.
    l_k: dict
        layers obtained by gflow algorithm. l_k[d] is a node set of depth d.
    """
    l_k = dict()
    g = dict()
    for node in graph.nodes:
        l_k[node] = 0
    return gflowaux(graph, input, output, meas_planes, 1, l_k, g, mode=mode)


def gflowaux(
    graph: nx.Graph,
    input: set[int],
    output: set[int],
    meas_planes: dict[int, str],
    k: int,
    l_k: dict[int, int],
    g: dict[int, set[int]],
    mode: str = "single",
):
    """Function to find one layer of the gflow.

    Ref: Backens et al., Quantum 5, 421 (2021).

    Parameters
    ----------
    graph: nx.Graph
        graph (incl. in and out)
    input: set
        set of node labels for input
    output: set
        set of node labels for output
    meas_planes: dict
        measurement planes for each qubits. meas_planes[i] is the measurement plane for qubit i.
    k: int
        current layer number.
    l_k: dict
        layers obtained by gflow algorithm. l_k[d] is a node set of depth d.
    g: dict
        gflow function. g[i] is the set of qubits to be corrected for the measurement of qubit i.
    mode: str(optional)
        The gflow finding algorithm can yield multiple equivalent solutions. So there are three options
            - "single": Returrns a single solution
            - "all": Returns all possible solutions
            - "abstract": Returns an abstract solution. Uncertainty is represented with sympy.Symbol objects,
              requiring user substitution to get a concrete answer.

    Returns
    -------
    g: dict
        gflow function. g[i] is the set of qubits to be corrected for the measurement of qubit i.
    l_k: dict
        layers obtained by gflow algorithm. l_k[d] is a node set of depth d.
    """

    nodes = set(graph.nodes)
    if output == nodes:
        return g, l_k
    non_output = nodes - output
    correction_candidate = output - input
    adj_mat, node_order_list = get_adjacency_matrix(graph)
    node_order_row = node_order_list.copy()
    node_order_row.sort()
    node_order_col = node_order_list.copy()
    node_order_col.sort()
    for out in output:
        adj_mat.remove_row(node_order_row.index(out))
        node_order_row.remove(out)
    adj_mat_row_reduced = adj_mat.copy()  # later use for construct RHS
    for node in nodes - correction_candidate:
        adj_mat.remove_col(node_order_col.index(node))
        node_order_col.remove(node)

    b = MatGF2(np.zeros((adj_mat.data.shape[0], len(non_output)), dtype=int))
    for i_row in range(len(node_order_row)):
        node = node_order_row[i_row]
        vec = MatGF2(np.zeros(len(node_order_row), dtype=int))
        if meas_planes[node] == "XY":
            vec.data[i_row] = 1
        elif meas_planes[node] == "ZX":
            vec.data[i_row] = 1
            vec_add = adj_mat_row_reduced.data[:, node_order_list.index(node)]
            vec = vec + vec_add
        elif meas_planes[node] == "YZ":
            vec.data = adj_mat_row_reduced.data[:, node_order_list.index(node)].reshape(vec.data.shape)
        b.data[:, i_row] = vec.data
    adj_mat, b, _, col_permutation = adj_mat.forward_eliminate(b)
    x, kernels = adj_mat.backward_substitute(b)

    corrected_nodes = set()
    for i_row in range(len(node_order_row)):
        non_out_node = node_order_row[i_row]
        x_col = x[:, i_row]
        if 0 in x_col.shape or x_col[0] == sp.nan:  # no solution
            continue
        if mode == "single":
            sol_list = [x_col[i].subs(zip(kernels, [sp.false] * len(kernels))) for i in range(len(x_col))]
            sol = np.array(sol_list)
            sol_index = sol.nonzero()[0]
            g[non_out_node] = set(node_order_col[col_permutation.index(i)] for i in sol_index)
            if meas_planes[non_out_node] in ["ZX", "YZ"]:
                g[non_out_node] |= {non_out_node}

        elif mode == "all":
            g[non_out_node] = set()
            binary_combinations = product([0, 1], repeat=len(kernels))
            for binary_combination in binary_combinations:
                sol_list = [x_col[i].subs(zip(kernels, binary_combination)) for i in range(len(x_col))]
                kernel_list = [True if i == 1 else False for i in binary_combination]
                sol_list.extend(kernel_list)
                sol = np.array(sol_list)
                sol_index = sol.nonzero()[0]
                g_i = set(node_order_col[col_permutation.index(i)] for i in sol_index)
                if meas_planes[non_out_node] in ["ZX", "YZ"]:
                    g_i |= {non_out_node}

                g[non_out_node] |= {frozenset(g_i)}

        elif mode == "abstract":
            g[non_out_node] = dict()
            for i in range(len(x_col)):
                node = node_order_col[col_permutation.index(i)]
                g[non_out_node][node] = x_col[i]
            for i in range(len(kernels)):
                g[non_out_node][node_order_col[col_permutation.index(len(x_col) + i)]] = kernels[i]
            if meas_planes[non_out_node] in ["ZX", "YZ"]:
                g[non_out_node][non_out_node] = sp.true

        l_k[non_out_node] = k
        corrected_nodes |= {non_out_node}

    if len(corrected_nodes) == 0:
        if output == nodes:
            return g, l_k
        else:
            return None, None
    else:
        return gflowaux(
            graph,
            input,
            output | corrected_nodes,
            meas_planes,
            k + 1,
            l_k,
            g,
            mode=mode,
        )


def find_flow(
    graph: nx.Graph,
    input: set[int],
    output: set[int],
    meas_planes: dict[int, str] = None,
) -> tuple[dict[int, set[int]], dict[int, int]]:
    """Causal flow finding algorithm

    For open graph g with input, output, and measurement planes, this returns causal flow.
    For more detail of causal flow, see Danos and Kashefi, PRA 74, 052310 (2006).

    Original algorithm by Mhalla and Perdrix,
    International Colloquium on Automata, Languages, and Programming (2008),
    pp. 857-868.

    Parameters
    ----------
    graph: nx.Graph
        graph (incl. in and out)
    input: set
        set of node labels for input
    output: set
        set of node labels for output
    meas_planes: int(optional)
        measurement planes for each qubits. meas_planes[i] is the measurement plane for qubit i.
        Note that an underlying graph has a causal flow only if all measurement planes are "XY".
        If not specified, all measurement planes are interpreted as "XY".

    Returns
    -------
    f: list of nodes
        causal flow function. f[i] is the qubit to be measured after qubit i.
    l_k: dict
        layers obtained by gflow algorithm. l_k[d] is a node set of depth d.
    """
    nodes = set(graph.nodes)
    edges = set(graph.edges)

    if meas_planes is None:
        meas_planes = {i: "XY" for i in (nodes - output)}

    for plane in meas_planes.values():
        if plane not in ["X", "Y", "XY"]:
            return None, None

    l_k = {i: 0 for i in nodes}
    f = dict()
    k = 1
    v_c = output - input
    return flowaux(nodes, edges, input, output, v_c, f, l_k, k)


def flowaux(
    nodes: set[int],
    edges: set[tuple[int, int]],
    input: set[int],
    output: set[int],
    v_c: set[int],
    f: dict[int, set[int]],
    l_k: dict[int, int],
    k: int,
):
    """Function to find one layer of the flow.

    Ref: Mhalla and Perdrix, International Colloquium on Automata,
    Languages, and Programming (Springer, 2008), pp. 857-868.

    Parameters
    ----------
    nodes: set
        labels of all qubits (nodes)
    edges: set
        edges
    input: set
        set of node labels for input
    output: set
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
        N = search_neighbor(q, edges)
        p_set = N & (nodes - output)
        if len(p_set) == 1:
            p = list(p_set)[0]
            f[p] = {q}
            l_k[p] = k
            v_out_prime = v_out_prime | {p}
            c_prime = c_prime | {q}
    # determine whether there exists flow
    if not v_out_prime:
        if output == nodes:
            return f, l_k
        else:
            return None, None
    return flowaux(
        nodes,
        edges,
        input,
        output | v_out_prime,
        (v_c - c_prime) | (v_out_prime & (nodes - input)),
        f,
        l_k,
        k + 1,
    )


def flow_from_pattern(pattern: Pattern) -> tuple[dict[int, set[int]], dict[int, int]]:
    """Check if the pattern has a valid flow. If so, return the flow and layers.

    Parameters
    ----------
    pattern: graphix.Pattern object
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
        if plane not in ["X", "Y", "XY"]:
            return None, None
    G = nx.Graph()
    nodes, edges = pattern.get_graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    input_nodes = pattern.input_nodes if not pattern.input_nodes else set()
    output_nodes = set(pattern.output_nodes)
    nodes = set(nodes)

    layers = pattern.get_layers()
    l_k = dict()
    for l in layers[1].keys():
        for n in layers[1][l]:
            l_k[n] = l
    lmax = max(l_k.values()) if l_k else 0
    for node in l_k.keys():
        l_k[node] = lmax - l_k[node] + 1
    for output_node in pattern.output_nodes:
        l_k[output_node] = 0

    xflow, zflow = get_corrections_from_pattern(pattern)

    if verify_flow(G, input_nodes, output_nodes, xflow):  # if xflow is valid
        zflow_from_xflow = dict()
        for node, corrections in deepcopy(xflow).items():
            cand = find_odd_neighbor(G, corrections) - {node}
            if cand:
                zflow_from_xflow[node] = cand
        if zflow_from_xflow != zflow:  # if zflow is consistent with xflow
            return None, None
        return xflow, l_k
    else:
        return None, None


def gflow_from_pattern(pattern: Pattern) -> tuple[dict[int, set[int]], dict[int, int]]:
    """Check if the pattern has a valid gflow. If so, return the gflow and layers.

    Parameters
    ----------
    pattern: graphix.Pattern object
        pattern to be based on

    Returns
    -------
    g: dict
        gflow function. g[i] is the set of qubits to be corrected for the measurement of qubit i.
    l_k: dict
        layers obtained by gflow algorithm. l_k[d] is a node set of depth d.
    """
    G = nx.Graph()
    nodes, edges = pattern.get_graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    input_nodes = set(pattern.input_nodes) if pattern.input_nodes else set()
    output_nodes = set(pattern.output_nodes)
    meas_planes = pattern.get_meas_plane()
    nodes = set(nodes)

    layers = pattern.get_layers()
    l_k = dict()
    for l in layers[1].keys():
        for n in layers[1][l]:
            l_k[n] = l
    lmax = max(l_k.values()) if l_k else 0
    for node in l_k.keys():
        l_k[node] = lmax - l_k[node] + 1
    for output_node in pattern.output_nodes:
        l_k[output_node] = 0

    xflow, zflow = get_corrections_from_pattern(pattern)
    for node, plane in meas_planes.items():
        if plane in ["ZX", "YZ"]:
            if node not in xflow.keys():
                xflow[node] = {node}
            xflow[node] |= {node}

    if verify_gflow(G, input_nodes, output_nodes, xflow, meas_planes):  # if xflow is valid
        zflow_from_xflow = dict()
        for node, corrections in deepcopy(xflow).items():
            cand = find_odd_neighbor(G, corrections) - {node}
            if cand:
                zflow_from_xflow[node] = cand
        if zflow_from_xflow != zflow:  # if zflow is consistent with xflow
            return None, None
        return xflow, l_k
    else:
        return None, None


def get_corrections_from_pattern(pattern: Pattern) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """Get x and z corrections from pattern

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
    xflow = dict()
    zflow = dict()
    for cmd in pattern:
        if cmd[0] == "M":
            target = cmd[1]
            xflow_source = {x for x in cmd[4] if cmd[4].count(x) % 2 != 0} & nodes
            zflow_source = {x for x in cmd[5] if cmd[5].count(x) % 2 != 0} & nodes
            for node in xflow_source:
                if node not in xflow.keys():
                    xflow[node] = set()
                xflow[node] |= {target}
            for node in zflow_source:
                if node not in zflow.keys():
                    zflow[node] = set()
                zflow[node] |= {target}
        if cmd[0] == "X":
            target = cmd[1]
            xflow_source = {x for x in cmd[2] if cmd[2].count(x) % 2 != 0} & nodes
            for node in xflow_source:
                if node not in xflow.keys():
                    xflow[node] = set()
                xflow[node] |= {target}
        if cmd[0] == "Z":
            target = cmd[1]
            zflow_source = {x for x in cmd[2] if cmd[2].count(x) % 2 != 0} & nodes
            for node in zflow_source:
                if node not in zflow.keys():
                    zflow[node] = set()
                zflow[node] |= {target}
    return xflow, zflow


def search_neighbor(node: int, edges: set[tuple[int, int]]) -> set[int]:
    """Function to find neighborhood of node in edges. This is an ancillary method for `flowaux()`.

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
    N = set()
    for edge in edges:
        if node == edge[0]:
            N = N | {edge[1]}
        elif node == edge[1]:
            N = N | {edge[0]}
    return N


def get_min_depth(l_k: dict[int, int]) -> int:
    """get minimum depth of graph.

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


def find_odd_neighbor(graph: nx.Graph, vertices: set[int]) -> set[int]:
    """Returns the set containing the odd neighbor of a set of vertices.

    Parameters
    ----------
    graph : networkx.Graph
        underlying graph.
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


def get_layers(l_k: dict[int, int]) -> tuple[int, dict[int, set[int]]]:
    """get components of each layer.
    Parameters
    -------
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
    layers = {k: set() for k in range(d + 1)}
    for i in l_k.keys():
        layers[l_k[i]] |= {i}
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
    try:  # if inputs is not empty
        dependence_flow = {input: set() for input in inputs}
    except:
        dependence_flow = dict()
    # concatenate flow and odd_flow
    combined_flow = dict()
    for node, corrections in flow.items():
        combined_flow[node] = corrections | odd_flow[node]
    for node, corrections in combined_flow.items():
        for correction in corrections:
            if correction not in dependence_flow.keys():
                dependence_flow[correction] = set()
            dependence_flow[correction] |= {node}
    return dependence_flow


def get_layers_from_flow(
    flow: dict[int, set], odd_flow: dict[int, set], inputs: set[int], outputs: set[int]
) -> tuple[dict[int, set], int]:
    """Get layers from flow (incl. gflow).

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
    layers = dict()
    depth = 0
    dependence_flow = get_dependence_flow(inputs, odd_flow, flow)
    left_nodes = set(flow.keys())
    for output in outputs:
        if output in left_nodes:
            raise ValueError("Invalid flow")
    while True:
        layers[depth] = set()
        for node in left_nodes:
            if node not in dependence_flow.keys() or len(dependence_flow[node]) == 0:
                layers[depth] |= {node}
            elif dependence_flow[node] == {node}:
                layers[depth] |= {node}
        left_nodes -= layers[depth]
        for node in left_nodes:
            dependence_flow[node] -= layers[depth]
        if len(layers[depth]) == 0:
            if len(left_nodes) == 0:
                layers[depth] = outputs
                depth += 1
                break
            else:
                raise ValueError("Invalid flow")
        else:
            depth += 1
    return layers, depth


def get_adjacency_matrix(graph: nx.Graph) -> tuple[MatGF2, list[int]]:
    """Get adjacency matrix of the graph

    Returns
    -------
    adjacency_matrix: graphix.linalg.MatGF2
        adjacency matrix of the graph. the matrix is defined on GF(2) field.
    node_list: list
        ordered list of nodes. node_list[i] is the node label of i-th row/column of the adjacency matrix.

    """
    node_list = list(graph.nodes)
    node_list.sort()
    adjacency_matrix = nx.to_numpy_array(graph, nodelist=node_list)
    adjacency_matrix = MatGF2(adjacency_matrix.astype(int))
    return adjacency_matrix, node_list


def verify_flow(
    graph: nx.Graph,
    input: set[int],
    output: set[int],
    flow: dict[int, set],
    meas_planes: dict[int, str] = {},
) -> bool:
    """Check whether the flow is valid.

    Parameters
    ----------
    graph: nx.Graph
        graph (incl. in and out)
    flow: dict[int, set]
        flow function. flow[i] is the set of qubits to be corrected for the measurement of qubit i.
    meas_planes: dict[int, str](optional)
        measurement planes for each qubits. meas_planes[i] is the measurement plane for qubit i.


    Returns
    -------
    valid_flow: bool
        True if the flow is valid. False otherwise.
    """

    valid_flow = True
    non_outputs = set(graph.nodes) - output
    # if meas_planes is given, check whether all measurement planes are "XY"
    for node, plane in meas_planes.items():
        if plane != "XY" or node not in non_outputs:
            valid_flow = False
            return valid_flow

    odd_flow = {node: find_odd_neighbor(graph, corrections) for node, corrections in flow.items()}

    try:
        _, _ = get_layers_from_flow(flow, odd_flow, input, output)
    except ValueError:
        valid_flow = False
        return valid_flow
    # check if v ~ f(v) for each node
    edges = set(graph.edges)
    for node, correction in flow.items():
        if len(correction) > 1:
            valid_flow = False
            return valid_flow
        correction = list(correction)[0]
        if (node, correction) not in edges and (correction, node) not in edges:
            valid_flow = False
            return valid_flow
    return valid_flow


def verify_gflow(
    graph: nx.Graph,
    input: set[int],
    output: set[int],
    gflow: dict[int, set],
    meas_planes: dict[int, str],
) -> bool:
    """Check whether the gflow is valid.

    Parameters
    ----------
    graph: nx.Graph
        graph (incl. in and out)
    input: set
        set of node labels for input
    output: set
        set of node labels for output
    gflow: dict[int, set]
        gflow function. gflow[i] is the set of qubits to be corrected for the measurement of qubit i.
        .. seealso:: :func:`gflow.gflow`
    meas_planes: dict[int, str]
        measurement planes for each qubits. meas_planes[i] is the measurement plane for qubit i.

    Returns
    -------
    valid_gflow: bool
        True if the gflow is valid. False otherwise.
    """
    valid_gflow = True
    non_outputs = set(graph.nodes) - output
    odd_flow = dict()
    for non_output in non_outputs:
        if non_output not in gflow:
            gflow[non_output] = set()
            odd_flow[non_output] = set()
        else:
            odd_flow[non_output] = find_odd_neighbor(graph, gflow[non_output])

    try:
        _, _ = get_layers_from_flow(gflow, odd_flow, input, output)
    except ValueError:
        valid_flow = False
        return valid_flow

    # check for each measurement plane
    for node, plane in meas_planes.items():
        # index = node_order.index(node)
        if plane == "XY":
            valid_gflow &= (node not in gflow[node]) and (node in odd_flow[node])
        elif plane == "XZ":
            valid_gflow &= (node in gflow[node]) and (node in odd_flow[node])
        elif plane == "YZ":
            valid_gflow &= (node in gflow[node]) and (node not in odd_flow[node])

    return valid_gflow


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
    inputs = non_output
    return inputs


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
    outputs = non_inputs - non_outputs
    return outputs
