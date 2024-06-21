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

from copy import deepcopy
from itertools import product

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
        elif meas_planes[node] == "XZ":
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
            if meas_planes[non_out_node] in ["XZ", "YZ"]:
                g[non_out_node] |= {non_out_node}

        elif mode == "all":
            g[non_out_node] = set()
            binary_combinations = product([0, 1], repeat=len(kernels))
            for binary_combination in binary_combinations:
                sol_list = [x_col[i].subs(zip(kernels, binary_combination)) for i in range(len(x_col))]
                sol = np.array(sol_list)
                sol_index = sol.nonzero()[0]
                g_i = set(node_order_col[col_permutation.index(i)] for i in sol_index)
                if meas_planes[non_out_node] in ["XZ", "YZ"]:
                    g_i |= {non_out_node}

                g[non_out_node] |= {frozenset(g_i)}

        elif mode == "abstract":
            g[non_out_node] = dict()
            for i in range(len(x_col)):
                node = node_order_col[col_permutation.index(i)]
                g[non_out_node][node] = x_col[i]
            if meas_planes[non_out_node] in ["XZ", "YZ"]:
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
        if plane != "XY":
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


def find_pauliflow(
    graph: nx.Graph,
    input: set[int],
    output: set[int],
    meas_planes: dict[int, str],
    meas_angles: dict[int, float],
    mode: str = "single",
) -> tuple[dict[int, set[int]], dict[int, int]]:
    """Maximally delayed Pauli flow finding algorithm

    For open graph g with input, output, measurement planes and measurement angles, this returns maximally delayed Pauli flow.

    Pauli flow consist of function p(i) where i is the qubit labels,
    and strict partial ordering < or layers labels l_k where each element
    specify the order of qubits to be measured to maintain determinism in MBQC.
    In practice, we must measure qubits in order specified in array l_k (increasing order
    of l_k from 1), and for each measurements of qubit i we must perform corrections on
    qubits in p(i), depending on the measurement outcome.

    For more details of Pauli flow and the finding algorithm used in this method,
    see Simmons et al., EPTCS 343, 2021, pp. 50-101 (arXiv:2109.05654).

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
    meas_angles: dict
        measurement angles for each qubits. meas_angles[i] is the measurement angle for qubit i.
    mode: str(optional)
        The Pauliflow finding algorithm can yield multiple equivalent solutions. So there are three options

        - "single": Returrns a single solution

        - "all": Returns all possible solutions

        - "abstract": Returns an abstract solution. Uncertainty is represented with sympy.Symbol objects,
          requiring user substitution to get a concrete answer.

        Default is "single".

    Returns
    -------
    p: dict
        Pauli flow function. p[i] is the set of qubits to be corrected for the measurement of qubit i.
    l_k: dict
        layers obtained by  Pauli flow algorithm. l_k[d] is a node set of depth d.
    """
    l_k = dict()
    p = dict()
    Lx, Ly, Lz = get_pauli_nodes(meas_planes, meas_angles)
    for node in graph.nodes:
        if node in output:
            l_k[node] = 0

    return pauliflowaux(graph, input, output, meas_planes, 0, set(), output, l_k, p, (Lx, Ly, Lz), mode)


def pauliflowaux(
    graph: nx.Graph,
    input: set[int],
    output: set[int],
    meas_planes: dict[int, str],
    k: int,
    correction_candidate: set[int],
    solved_nodes: set[int],
    l_k: dict[int, int],
    p: dict[int, set[int]],
    L: tuple[set[int], set[int], set[int]],
    mode: str = "single",
):
    """Function to find one layer of the Pauli flow.

    Ref: Simmons et al., EPTCS 343, 2021, pp. 50-101 (arXiv:2109.05654).

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
    correction_candidate: set
        set of qubits to be corrected.
    solved_nodes: set
        set of qubits whose layers are already determined.
    l_k: dict
        layers obtained by gflow algorithm. l_k[d] is a node set of depth d.
    p: dict
        Pauli flow function. p[i] is the set of qubits to be corrected for the measurement of qubit i.
    L: tuple
        L = (Lx, Ly, Lz) where Lx, Ly, Lz are sets of qubits whose measurement operators are X, Y, Z, respectively.
    mode: str(optional)
        The Pauliflow finding algorithm can yield multiple equivalent solutions. So there are three options
          - "single": Returrns a single solution
          - "all": Returns all possible solutions
          - "abstract": Returns an abstract solution. Uncertainty is represented with sympy.Symbol objects,
            requiring user substitution to get a concrete answer.

    Returns
    -------
    p: dict
        Pauli flow function. p[i] is the set of qubits to be corrected for the measurement of qubit i.
    l_k: dict
        layers obtained by Pauli flow algorithm. l_k[d] is a node set of depth d.
    """
    Lx, Ly, Lz = L
    solved_update = set()
    nodes = set(graph.nodes)
    if output == nodes:
        return p, l_k
    unsolved_nodes = nodes - solved_nodes

    adj_mat, node_order_list = get_adjacency_matrix(graph)
    adj_mat_w_id = adj_mat.copy() + MatGF2(np.identity(adj_mat.data.shape[0], dtype=int))
    node_order_row = node_order_list.copy()
    node_order_row_lower = node_order_list.copy()
    node_order_col = node_order_list.copy()

    Pbar = correction_candidate | Ly | Lz
    P = nodes - Pbar
    K = (correction_candidate | Lx | Ly) & (nodes - input)
    Y = Ly - correction_candidate

    for node in unsolved_nodes:
        adj_mat_ = adj_mat.copy()
        adj_mat_w_id_ = adj_mat_w_id.copy()
        node_order_row_ = node_order_row.copy()
        node_order_row_lower_ = node_order_row_lower.copy()
        node_order_col_ = node_order_col.copy()
        for node_ in nodes - (P | {node}):
            adj_mat_.remove_row(node_order_row_.index(node_))
            node_order_row_.remove(node_)
        for node_ in nodes - (Y - {node}):
            adj_mat_w_id_.remove_row(node_order_row_lower_.index(node_))
            node_order_row_lower_.remove(node_)
        for node_ in nodes - (K - {node}):
            adj_mat_.remove_col(node_order_col_.index(node_))
            adj_mat_w_id_.remove_col(node_order_col_.index(node_))
            node_order_col_.remove(node_)
        adj_mat_.concatenate(adj_mat_w_id_, axis=0)

        if mode == "all":
            p[node] = set()

        if mode == "abstract":
            p[node] = list()

        solved = False
        if meas_planes[node] == "XY" or node in Lx or node in Ly:
            S = MatGF2(np.zeros((len(node_order_row_), 1), dtype=int))
            S.data[node_order_row_.index(node), :] = 1
            S_lower = MatGF2(np.zeros((len(node_order_row_lower_), 1), dtype=int))
            S.concatenate(S_lower, axis=0)
            adj_mat_XY, S, _, col_permutation_XY = adj_mat_.forward_eliminate(S, copy=True)
            x_XY, kernels = adj_mat_XY.backward_substitute(S)

            if 0 not in x_XY.shape and x_XY[0, 0] != sp.nan:
                solved_update |= {node}
                x_XY = x_XY[:, 0]
                l_k[node] = k

                if mode == "single":
                    sol_list = [x_XY[i].subs(zip(kernels, [sp.false] * len(kernels))) for i in range(len(x_XY))]
                    sol = np.array(sol_list)
                    sol_index = sol.nonzero()[0]
                    p[node] = set(node_order_col_[col_permutation_XY.index(i)] for i in sol_index)
                    solved = True

                elif mode == "all":
                    binary_combinations = product([0, 1], repeat=len(kernels))
                    for binary_combination in binary_combinations:
                        sol_list = [x_XY[i].subs(zip(kernels, binary_combination)) for i in range(len(x_XY))]
                        sol = np.array(sol_list)
                        sol_index = sol.nonzero()[0]
                        p_i = set(node_order_col_[col_permutation_XY.index(i)] for i in sol_index)
                        p[node].add(frozenset(p_i))

                elif mode == "abstract":
                    p_i = dict()
                    for i in range(len(x_XY)):
                        node_temp = node_order_col_[col_permutation_XY.index(i)]
                        p_i[node_temp] = x_XY[i]
                    p[node].append(p_i)

        if not solved and (meas_planes[node] == "XZ" or node in Lz or node in Lx):
            S = MatGF2(np.zeros((len(node_order_row_), 1), dtype=int))
            S.data[node_order_row_.index(node)] = 1
            for neighbor in search_neighbor(node, graph.edges):
                if neighbor in P | {node}:
                    S.data[node_order_row_.index(neighbor), :] = 1
            S_lower = MatGF2(np.zeros((len(node_order_row_lower_), 1), dtype=int))
            for neighbor in search_neighbor(node, graph.edges):
                if neighbor in Y - {node}:
                    S_lower.data[node_order_row_lower_.index(neighbor), :] = 1
            S.concatenate(S_lower, axis=0)
            adj_mat_XZ, S, _, col_permutation_XZ = adj_mat_.forward_eliminate(S, copy=True)
            x_XZ, kernels = adj_mat_XZ.backward_substitute(S)
            if 0 not in x_XZ.shape and x_XZ[0, 0] != sp.nan:
                solved_update |= {node}
                x_XZ = x_XZ[:, 0]
                l_k[node] = k

                if mode == "single":
                    sol_list = [x_XZ[i].subs(zip(kernels, [sp.false] * len(kernels))) for i in range(len(x_XZ))]
                    sol = np.array(sol_list)
                    sol_index = sol.nonzero()[0]
                    p[node] = set(node_order_col_[col_permutation_XZ.index(i)] for i in sol_index) | {node}
                    solved = True

                elif mode == "all":
                    binary_combinations = product([0, 1], repeat=len(kernels))
                    for binary_combination in binary_combinations:
                        sol_list = [x_XZ[i].subs(zip(kernels, binary_combination)) for i in range(len(x_XZ))]
                        sol = np.array(sol_list)
                        sol_index = sol.nonzero()[0]
                        p_i = set(node_order_col_[col_permutation_XZ.index(i)] for i in sol_index) | {node}
                        p[node].add(frozenset(p_i))

                elif mode == "abstract":
                    p_i = dict()
                    for i in range(len(x_XZ)):
                        node_temp = node_order_col_[col_permutation_XZ.index(i)]
                        p_i[node_temp] = x_XZ[i]
                    p_i[node] = sp.true
                    p[node].append(p_i)

        if not solved and (meas_planes[node] == "YZ" or node in Ly or node in Lz):
            S = MatGF2(np.zeros((len(node_order_row_), 1), dtype=int))
            for neighbor in search_neighbor(node, graph.edges):
                if neighbor in P | {node}:
                    S.data[node_order_row_.index(neighbor), :] = 1
            S_lower = MatGF2(np.zeros((len(node_order_row_lower_), 1), dtype=int))
            for neighbor in search_neighbor(node, graph.edges):
                if neighbor in Y - {node}:
                    S_lower.data[node_order_row_lower_.index(neighbor), :] = 1
            S.concatenate(S_lower, axis=0)
            adj_mat_YZ, S, _, col_permutation_YZ = adj_mat_.forward_eliminate(S, copy=True)
            x_YZ, kernels = adj_mat_YZ.backward_substitute(S)
            if 0 not in x_YZ.shape and x_YZ[0, 0] != sp.nan:
                solved_update |= {node}
                x_YZ = x_YZ[:, 0]
                l_k[node] = k

                if mode == "single":
                    sol_list = [x_YZ[i].subs(zip(kernels, [sp.false] * len(kernels))) for i in range(len(x_YZ))]
                    sol = np.array(sol_list)
                    sol_index = sol.nonzero()[0]
                    p[node] = set(node_order_col_[col_permutation_YZ.index(i)] for i in sol_index) | {node}
                    solved = True

                elif mode == "all":
                    binary_combinations = product([0, 1], repeat=len(kernels))
                    for binary_combination in binary_combinations:
                        sol_list = [x_YZ[i].subs(zip(kernels, binary_combination)) for i in range(len(x_YZ))]
                        sol = np.array(sol_list)
                        sol_index = sol.nonzero()[0]
                        p_i = set(node_order_col_[col_permutation_YZ.index(i)] for i in sol_index) | {node}
                        p[node].add(frozenset(p_i))

                elif mode == "abstract":
                    p_i = dict()
                    for i in range(len(x_YZ)):
                        node_temp = node_order_col_[col_permutation_YZ.index(i)]
                        p_i[node_temp] = x_YZ[i]
                    p_i[node] = sp.true
                    p[node].append(p_i)

    if solved_update == set() and k > 0:
        if solved_nodes == nodes:
            return p, l_k
        else:
            return None, None
    else:
        B = solved_nodes | solved_update
        return pauliflowaux(graph, input, output, meas_planes, k + 1, B, B, l_k, p, (Lx, Ly, Lz), mode)


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
        if plane != "XY":
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
        if plane in ["XZ", "YZ"]:
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


def pauliflow_from_pattern(pattern: Pattern, mode="single") -> tuple[dict[int, set[int]], dict[int, int]]:
    """Check if the pattern has a valid Pauliflow. If so, return the Pauliflow and layers.

    Parameters
    ----------
    pattern: graphix.Pattern object
        pattern to be based on
    mode: str(optional)
        The Pauliflow finding algorithm can yield multiple equivalent solutions. So there are two options
            - "single": Returrns a single solution
            - "all": Returns all possible solutions
    Returns
    -------
    p: dict
        Pauli flow function. p[i] is the set of qubits to be corrected for the measurement of qubit i.
    l_k: dict
        layers obtained by Pauli flow algorithm. l_k[d] is a node set of depth d.
    """
    G = nx.Graph()
    nodes, edges = pattern.get_graph()
    nodes = set(nodes)
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    input_nodes = set(pattern.input_nodes) if pattern.input_nodes else set()
    output_nodes = set(pattern.output_nodes)
    non_outputs = nodes - output_nodes
    meas_planes = pattern.get_meas_plane()
    meas_angles = pattern.get_angles()
    nodes = set(nodes)

    Lx, Ly, Lz = get_pauli_nodes(meas_planes, meas_angles)

    p_all, l_k = find_pauliflow(G, input_nodes, output_nodes, meas_planes, meas_angles, mode="all")
    if p_all is None:
        return None, None

    p = dict()

    xflow, zflow = get_corrections_from_pattern(pattern)
    for node in non_outputs:
        xflow_node = xflow[node] if node in xflow.keys() else set()
        zflow_node = zflow[node] if node in zflow.keys() else set()
        p_list = list(p_all[node]) if node in p_all.keys() else []
        valid = False

        for p_i in p_list:
            if xflow_node & p_i == xflow_node:
                ignored_nodes = p_i - xflow_node - {node}
                # check if nodes in ignored_nodes are measured in X or Y basis
                if ignored_nodes & (Lx | Ly) != ignored_nodes:
                    continue
                odd_neighbers = find_odd_neighbor(G, p_i)
                if zflow_node & odd_neighbers == zflow_node:
                    ignored_nodes = zflow_node - odd_neighbers - {node}
                    # check if nodes in ignored_nodes are measured in Z or Y basis
                    if ignored_nodes & (Ly | Lz) == ignored_nodes:
                        valid = True
                        if mode == "single":
                            p[node] = set(p_i)
                            break
                        elif mode == "all":
                            if node not in p.keys():
                                p[node] = set()
                            p[node].add(frozenset(p_i))
                            continue
        if not valid:
            return None, None

    return p, l_k


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
        if plane != "XY":
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
        if plane in ["XZ", "YZ"]:
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


def get_dependence_pauliflow(
    inputs: set[int],
    flow: dict[int, set[int]],
    odd_flow: dict[int, set[int]],
    L: tuple[set[int], set[int], set[int]],
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
    L: tuple
        L = (Lx, Ly, Lz) where Lx, Ly, Lz are sets of qubits whose measurement operators are X, Y, Z, respectively.

    Returns
    -------
    dependence_pauliflow: dict[int, set[int]]
        dependence flow function. dependence_pauliflow[i] is the set of qubits to be corrected for the measurement of qubit i.
    """
    Lx, Ly, Lz = L
    dependence_pauliflow = {input: set() for input in inputs}
    # concatenate p and odd_p
    combined_flow = dict()
    for node, corrections in flow.items():
        combined_flow[node] = (corrections - (Lx | Ly)) | (odd_flow[node] - (Ly | Lz))
        for ynode in Ly:
            if ynode in corrections.symmetric_difference(odd_flow[node]):
                combined_flow[node] |= {ynode}
    for node, corrections in combined_flow.items():
        for correction in corrections:
            if correction not in dependence_pauliflow.keys():
                dependence_pauliflow[correction] = set()
            dependence_pauliflow[correction] |= {node}
    return dependence_pauliflow


def get_layers_from_flow(
    flow: dict[int, set],
    odd_flow: dict[int, set],
    inputs: set[int],
    outputs: set[int],
    L: tuple[set[int], set[int], set[int]] | None = None,
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
    L: tuple
        L = (Lx, Ly, Lz) where Lx, Ly, Lz are sets of qubits whose measurement operators are X, Y, Z, respectively.
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
    layers = dict()
    depth = 0
    if L is None:
        dependence_flow = get_dependence_flow(inputs, odd_flow, flow)
    else:
        dependence_flow = get_dependence_pauliflow(inputs, flow, odd_flow, L)
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


def verify_pauliflow(
    graph: nx.Graph,
    input: set[int],
    output: set[int],
    pauliflow: dict[int, set[int]],
    meas_planes: dict[int, str],
    meas_angles: dict[int, float],
) -> bool:
    """Check whether the Pauliflow is valid.

    Parameters
    ----------
    graph: nx.Graph
        graph (incl. in and out)
    input: set
        set of node labels for input
    output: set
        set of node labels for output
    pauliflow: dict[int, set]
        Pauli flow function. pauliflow[i] is the set of qubits to be corrected for the measurement of qubit i.
    meas_planes: dict[int, str]
        measurement planes for each qubits. meas_planes[i] is the measurement plane for qubit i.
    meas_angles: dict[int, float]
        measurement angles for each qubits. meas_angles[i] is the measurement angle for qubit i.

    Returns
    -------
    valid_pauliflow: bool
        True if the Pauliflow is valid. False otherwise.
    """
    Lx, Ly, Lz = get_pauli_nodes(meas_planes, meas_angles)

    valid_pauliflow = True
    non_outputs = set(graph.nodes) - output
    odd_flow = dict()
    for non_output in non_outputs:
        if non_output not in pauliflow.keys():
            pauliflow[non_output] = set()
            odd_flow[non_output] = set()
        else:
            odd_flow[non_output] = find_odd_neighbor(graph, pauliflow[non_output])

    try:
        layers, depth = get_layers_from_flow(pauliflow, odd_flow, input, output, (Lx, Ly, Lz))
    except ValueError:
        valid_flow = False
        return valid_flow
    node_order = []
    for d in range(depth):
        node_order.extend(list(layers[d]))

    for node, plane in meas_planes.items():
        if node in Lx:
            valid_pauliflow &= node in odd_flow[node]
        elif node in Lz:
            valid_pauliflow &= node in pauliflow[node]
        elif node in Ly:
            valid_pauliflow &= node in pauliflow[node].symmetric_difference(odd_flow[node])
        elif plane == "XY":
            valid_pauliflow &= (node not in pauliflow[node]) and (node in odd_flow[node])
        elif plane == "XZ":
            valid_pauliflow &= (node in pauliflow[node]) and (node in odd_flow[node])
        elif plane == "YZ":
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


def get_pauli_nodes(meas_planes: dict[int, str], meas_angles: dict[int, float]) -> tuple[set[int], set[int], set[int]]:
    """Get sets of nodes measured in X, Y, Z basis.

    Parameters
    ----------
    meas_planes: dict[int, str]
        measurement planes for each node.
    meas_angles: dict[int, float]
        measurement angles for each node.

    Returns
    -------
    Lx: set
        set of nodes measured in X basis.
    Ly: set
        set of nodes measured in Y basis.
    Lz: set
        set of nodes measured in Z basis.
    """
    Lx, Ly, Lz = set(), set(), set()
    for node, plane in meas_planes.items():
        if plane == "XY" and meas_angles[node] == int(meas_angles[node]):  # measurement angle is integer
            Lx |= {node}
        elif plane == "XY" and 2 * meas_angles[node] == int(2 * meas_angles[node]):  # measurement angle is half integer
            Ly |= {node}
        elif plane == "XZ" and meas_angles[node] == int(meas_angles[node]):
            Lz |= {node}
        elif plane == "XZ" and 2 * meas_angles[node] == int(2 * meas_angles[node]):
            Lx |= {node}
        elif plane == "YZ" and meas_angles[node] == int(meas_angles[node]):
            Ly |= {node}
        elif plane == "YZ" and 2 * meas_angles[node] == int(2 * meas_angles[node]):
            Lz |= {node}
    return Lx, Ly, Lz
