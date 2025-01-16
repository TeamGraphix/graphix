"""Flow finding algorithm.

For a given underlying graph (G, I, O, meas_plane), this method finds a (generalized) flow [NJP 9, 250 (2007)]
in polynomincal time.
In particular, this outputs gflow with minimum depth, maximally delayed gflow.

Ref: Mhalla and Perdrix, International Colloquium on Automata,
Languages, and Programming (Springer, 2008), pp. 857-868.
Ref: Backens et al., Quantum 5, 421 (2021).

"""

from __future__ import annotations

import numbers
from copy import deepcopy
from itertools import product
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import sympy as sp

from graphix import utils
from graphix.command import CommandKind
from graphix.fundamentals import Plane
from graphix.linalg import MatGF2

if TYPE_CHECKING:
    from graphix.pattern import Pattern


# TODO: This should be ensured by type-checking.
def check_meas_planes(meas_planes: dict[int, Plane]) -> None:
    """Check that all planes are valid planes."""
    for node, plane in meas_planes.items():
        if not isinstance(plane, Plane):
            raise ValueError(f"Measure plane for {node} is `{plane}`, which is not an instance of `Plane`")


def find_gflow(
    graph: nx.Graph,
    iset: set[int],
    oset: set[int],
    meas_planes: dict[int, Plane],
    mode: str = "single",
) -> tuple[dict[int, set[int]], dict[int, int]]:
    """Maximally delayed gflow finding algorithm.

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
    iset: set
        set of node labels for input
    oset: set
        set of node labels for output
    meas_planes: dict
        measurement planes for each qubits. meas_planes[i] is the measurement plane for qubit i.
    mode: str
        The gflow finding algorithm can yield multiple equivalent solutions. So there are three options

        - "single": Returrns a single solution

        - "all": Returns all possible solutions

        - "abstract": Returns an abstract solution. Uncertainty is represented with sympy.Symbol objects,
          requiring user substitution to get a concrete answer.

        Optional. Default is "single".

    Returns
    -------
    g: dict
        gflow function. g[i] is the set of qubits to be corrected for the measurement of qubit i.
    l_k: dict
        layers obtained by gflow algorithm. l_k[d] is a node set of depth d.
    """
    check_meas_planes(meas_planes)
    l_k = dict()
    g = dict()
    for node in graph.nodes:
        l_k[node] = 0
    return gflowaux(graph, iset, oset, meas_planes, 1, l_k, g, mode=mode)


def gflowaux(
    graph: nx.Graph,
    iset: set[int],
    oset: set[int],
    meas_planes: dict[int, Plane],
    k: int,
    l_k: dict[int, int],
    g: dict[int, set[int]],
    mode: str = "single",
):
    """Find one layer of the gflow.

    Ref: Backens et al., Quantum 5, 421 (2021).

    Parameters
    ----------
    graph: nx.Graph
        graph (incl. in and out)
    iset: set
        set of node labels for input
    oset: set
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
    if oset == nodes:
        return g, l_k
    non_output = nodes - oset
    correction_candidate = oset - iset
    adj_mat, node_order_list = get_adjacency_matrix(graph)
    node_order_row = node_order_list.copy()
    node_order_row.sort()
    node_order_col = node_order_list.copy()
    node_order_col.sort()
    for out in oset:
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
        if meas_planes[node] == Plane.XY:
            vec.data[i_row] = 1
        elif meas_planes[node] == Plane.XZ:
            vec.data[i_row] = 1
            vec_add = adj_mat_row_reduced.data[:, node_order_list.index(node)]
            vec = vec + vec_add
        elif meas_planes[node] == Plane.YZ:
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
            if meas_planes[non_out_node] in [Plane.XZ, Plane.YZ]:
                g[non_out_node] |= {non_out_node}

        elif mode == "all":
            g[non_out_node] = set()
            binary_combinations = product([0, 1], repeat=len(kernels))
            for binary_combination in binary_combinations:
                sol_list = [x_col[i].subs(zip(kernels, binary_combination)) for i in range(len(x_col))]
                sol = np.array(sol_list)
                sol_index = sol.nonzero()[0]
                g_i = set(node_order_col[col_permutation.index(i)] for i in sol_index)
                if meas_planes[non_out_node] in [Plane.XZ, Plane.YZ]:
                    g_i |= {non_out_node}

                g[non_out_node] |= {frozenset(g_i)}

        elif mode == "abstract":
            g[non_out_node] = dict()
            for i in range(len(x_col)):
                node = node_order_col[col_permutation.index(i)]
                g[non_out_node][node] = x_col[i]
            if meas_planes[non_out_node] in [Plane.XZ, Plane.YZ]:
                g[non_out_node][non_out_node] = sp.true

        l_k[non_out_node] = k
        corrected_nodes |= {non_out_node}

    if len(corrected_nodes) == 0:
        if oset == nodes:
            return g, l_k
        else:
            return None, None
    else:
        return gflowaux(
            graph,
            iset,
            oset | corrected_nodes,
            meas_planes,
            k + 1,
            l_k,
            g,
            mode=mode,
        )


def find_flow(
    graph: nx.Graph,
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
    graph: nx.Graph
        graph (incl. in and out)
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
        meas_planes = {i: Plane.XY for i in (nodes - oset)}

    for plane in meas_planes.values():
        if plane != Plane.XY:
            return None, None

    l_k = {i: 0 for i in nodes}
    f = dict()
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
            v_out_prime = v_out_prime | {p}
            c_prime = c_prime | {q}
    # determine whether there exists flow
    if not v_out_prime:
        if oset == nodes:
            return f, l_k
        else:
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


def find_pauliflow(
    graph: nx.Graph,
    iset: set[int],
    oset: set[int],
    meas_planes: dict[int, Plane],
    meas_angles: dict[int, float],
    mode: str = "single",
) -> tuple[dict[int, set[int]], dict[int, int]]:
    """Maximally delayed Pauli flow finding algorithm.

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
    iset: set
        set of node labels for input
    oset: set
        set of node labels for output
    meas_planes: dict
        measurement planes for each qubits. meas_planes[i] is the measurement plane for qubit i.
    meas_angles: dict
        measurement angles for each qubits. meas_angles[i] is the measurement angle for qubit i.
    mode: str
        The Pauliflow finding algorithm can yield multiple equivalent solutions. So there are three options

        - "single": Returrns a single solution

        - "all": Returns all possible solutions

        - "abstract": Returns an abstract solution. Uncertainty is represented with sympy.Symbol objects,
          requiring user substitution to get a concrete answer.

        Optional. Default is "single".

    Returns
    -------
    p: dict
        Pauli flow function. p[i] is the set of qubits to be corrected for the measurement of qubit i.
    l_k: dict
        layers obtained by  Pauli flow algorithm. l_k[d] is a node set of depth d.
    """
    check_meas_planes(meas_planes)
    l_k = dict()
    p = dict()
    l_x, l_y, l_z = get_pauli_nodes(meas_planes, meas_angles)
    for node in graph.nodes:
        if node in oset:
            l_k[node] = 0

    return pauliflowaux(graph, iset, oset, meas_planes, 0, set(), oset, l_k, p, (l_x, l_y, l_z), mode)


def pauliflowaux(
    graph: nx.Graph,
    iset: set[int],
    oset: set[int],
    meas_planes: dict[int, Plane],
    k: int,
    correction_candidate: set[int],
    solved_nodes: set[int],
    l_k: dict[int, int],
    p: dict[int, set[int]],
    ls: tuple[set[int], set[int], set[int]],
    mode: str = "single",
):
    """Find one layer of the Pauli flow.

    Ref: Simmons et al., EPTCS 343, 2021, pp. 50-101 (arXiv:2109.05654).

    Parameters
    ----------
    graph: nx.Graph
        graph (incl. in and out)
    iset: set
        set of node labels for input
    oset: set
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
    ls: tuple
        ls = (l_x, l_y, l_z) where l_x, l_y, l_z are sets of qubits whose measurement operators are X, Y, Z, respectively.
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
    l_x, l_y, l_z = ls
    solved_update = set()
    nodes = set(graph.nodes)
    if oset == nodes:
        return p, l_k
    unsolved_nodes = nodes - solved_nodes

    adj_mat, node_order_list = get_adjacency_matrix(graph)
    adj_mat_w_id = adj_mat.copy() + MatGF2(np.identity(adj_mat.data.shape[0], dtype=int))
    node_order_row = node_order_list.copy()
    node_order_row_lower = node_order_list.copy()
    node_order_col = node_order_list.copy()

    p_bar = correction_candidate | l_y | l_z
    pset = nodes - p_bar
    kset = (correction_candidate | l_x | l_y) & (nodes - iset)
    yset = l_y - correction_candidate

    for node in unsolved_nodes:
        adj_mat_ = adj_mat.copy()
        adj_mat_w_id_ = adj_mat_w_id.copy()
        node_order_row_ = node_order_row.copy()
        node_order_row_lower_ = node_order_row_lower.copy()
        node_order_col_ = node_order_col.copy()
        for node_ in nodes - (pset | {node}):
            adj_mat_.remove_row(node_order_row_.index(node_))
            node_order_row_.remove(node_)
        for node_ in nodes - (yset - {node}):
            adj_mat_w_id_.remove_row(node_order_row_lower_.index(node_))
            node_order_row_lower_.remove(node_)
        for node_ in nodes - (kset - {node}):
            adj_mat_.remove_col(node_order_col_.index(node_))
            adj_mat_w_id_.remove_col(node_order_col_.index(node_))
            node_order_col_.remove(node_)
        adj_mat_.concatenate(adj_mat_w_id_, axis=0)

        if mode == "all":
            p[node] = set()

        if mode == "abstract":
            p[node] = list()

        solved = False
        if meas_planes[node] == Plane.XY or node in l_x or node in l_y:
            mat = MatGF2(np.zeros((len(node_order_row_), 1), dtype=int))
            mat.data[node_order_row_.index(node), :] = 1
            mat_lower = MatGF2(np.zeros((len(node_order_row_lower_), 1), dtype=int))
            mat.concatenate(mat_lower, axis=0)
            adj_mat_xy, mat, _, col_permutation_xy = adj_mat_.forward_eliminate(mat, copy=True)
            x_xy, kernels = adj_mat_xy.backward_substitute(mat)

            if 0 not in x_xy.shape and x_xy[0, 0] != sp.nan:
                solved_update |= {node}
                x_xy = x_xy[:, 0]
                l_k[node] = k

                if mode == "single":
                    sol_list = [x_xy[i].subs(zip(kernels, [sp.false] * len(kernels))) for i in range(len(x_xy))]
                    sol = np.array(sol_list)
                    sol_index = sol.nonzero()[0]
                    p[node] = set(node_order_col_[col_permutation_xy.index(i)] for i in sol_index)
                    solved = True

                elif mode == "all":
                    binary_combinations = product([0, 1], repeat=len(kernels))
                    for binary_combination in binary_combinations:
                        sol_list = [x_xy[i].subs(zip(kernels, binary_combination)) for i in range(len(x_xy))]
                        sol = np.array(sol_list)
                        sol_index = sol.nonzero()[0]
                        p_i = set(node_order_col_[col_permutation_xy.index(i)] for i in sol_index)
                        p[node].add(frozenset(p_i))

                elif mode == "abstract":
                    p_i = dict()
                    for i in range(len(x_xy)):
                        node_temp = node_order_col_[col_permutation_xy.index(i)]
                        p_i[node_temp] = x_xy[i]
                    p[node].append(p_i)

        if not solved and (meas_planes[node] == Plane.XZ or node in l_z or node in l_x):
            mat = MatGF2(np.zeros((len(node_order_row_), 1), dtype=int))
            mat.data[node_order_row_.index(node)] = 1
            for neighbor in search_neighbor(node, graph.edges):
                if neighbor in pset | {node}:
                    mat.data[node_order_row_.index(neighbor), :] = 1
            mat_lower = MatGF2(np.zeros((len(node_order_row_lower_), 1), dtype=int))
            for neighbor in search_neighbor(node, graph.edges):
                if neighbor in yset - {node}:
                    mat_lower.data[node_order_row_lower_.index(neighbor), :] = 1
            mat.concatenate(mat_lower, axis=0)
            adj_mat_xz, mat, _, col_permutation_xz = adj_mat_.forward_eliminate(mat, copy=True)
            x_xz, kernels = adj_mat_xz.backward_substitute(mat)
            if 0 not in x_xz.shape and x_xz[0, 0] != sp.nan:
                solved_update |= {node}
                x_xz = x_xz[:, 0]
                l_k[node] = k

                if mode == "single":
                    sol_list = [x_xz[i].subs(zip(kernels, [sp.false] * len(kernels))) for i in range(len(x_xz))]
                    sol = np.array(sol_list)
                    sol_index = sol.nonzero()[0]
                    p[node] = set(node_order_col_[col_permutation_xz.index(i)] for i in sol_index) | {node}
                    solved = True

                elif mode == "all":
                    binary_combinations = product([0, 1], repeat=len(kernels))
                    for binary_combination in binary_combinations:
                        sol_list = [x_xz[i].subs(zip(kernels, binary_combination)) for i in range(len(x_xz))]
                        sol = np.array(sol_list)
                        sol_index = sol.nonzero()[0]
                        p_i = set(node_order_col_[col_permutation_xz.index(i)] for i in sol_index) | {node}
                        p[node].add(frozenset(p_i))

                elif mode == "abstract":
                    p_i = dict()
                    for i in range(len(x_xz)):
                        node_temp = node_order_col_[col_permutation_xz.index(i)]
                        p_i[node_temp] = x_xz[i]
                    p_i[node] = sp.true
                    p[node].append(p_i)

        if not solved and (meas_planes[node] == Plane.YZ or node in l_y or node in l_z):
            mat = MatGF2(np.zeros((len(node_order_row_), 1), dtype=int))
            for neighbor in search_neighbor(node, graph.edges):
                if neighbor in pset | {node}:
                    mat.data[node_order_row_.index(neighbor), :] = 1
            mat_lower = MatGF2(np.zeros((len(node_order_row_lower_), 1), dtype=int))
            for neighbor in search_neighbor(node, graph.edges):
                if neighbor in yset - {node}:
                    mat_lower.data[node_order_row_lower_.index(neighbor), :] = 1
            mat.concatenate(mat_lower, axis=0)
            adj_mat_yz, mat, _, col_permutation_yz = adj_mat_.forward_eliminate(mat, copy=True)
            x_yz, kernels = adj_mat_yz.backward_substitute(mat)
            if 0 not in x_yz.shape and x_yz[0, 0] != sp.nan:
                solved_update |= {node}
                x_yz = x_yz[:, 0]
                l_k[node] = k

                if mode == "single":
                    sol_list = [x_yz[i].subs(zip(kernels, [sp.false] * len(kernels))) for i in range(len(x_yz))]
                    sol = np.array(sol_list)
                    sol_index = sol.nonzero()[0]
                    p[node] = set(node_order_col_[col_permutation_yz.index(i)] for i in sol_index) | {node}
                    solved = True

                elif mode == "all":
                    binary_combinations = product([0, 1], repeat=len(kernels))
                    for binary_combination in binary_combinations:
                        sol_list = [x_yz[i].subs(zip(kernels, binary_combination)) for i in range(len(x_yz))]
                        sol = np.array(sol_list)
                        sol_index = sol.nonzero()[0]
                        p_i = set(node_order_col_[col_permutation_yz.index(i)] for i in sol_index) | {node}
                        p[node].add(frozenset(p_i))

                elif mode == "abstract":
                    p_i = dict()
                    for i in range(len(x_yz)):
                        node_temp = node_order_col_[col_permutation_yz.index(i)]
                        p_i[node_temp] = x_yz[i]
                    p_i[node] = sp.true
                    p[node].append(p_i)

    if solved_update == set() and k > 0:
        if solved_nodes == nodes:
            return p, l_k
        else:
            return None, None
    else:
        bset = solved_nodes | solved_update
        return pauliflowaux(graph, iset, oset, meas_planes, k + 1, bset, bset, l_k, p, (l_x, l_y, l_z), mode)


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

    if verify_flow(g, input_nodes, output_nodes, xflow):  # if xflow is valid
        zflow_from_xflow = dict()
        for node, corrections in deepcopy(xflow).items():
            cand = find_odd_neighbor(g, corrections) - {node}
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
        if plane in [Plane.XZ, Plane.YZ]:
            if node not in xflow.keys():
                xflow[node] = {node}
            xflow[node] |= {node}

    if verify_gflow(g, input_nodes, output_nodes, xflow, meas_planes):  # if xflow is valid
        zflow_from_xflow = dict()
        for node, corrections in deepcopy(xflow).items():
            cand = find_odd_neighbor(g, corrections) - {node}
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
    g = nx.Graph()
    nodes, edges = pattern.get_graph()
    nodes = set(nodes)
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    input_nodes = set(pattern.input_nodes) if pattern.input_nodes else set()
    output_nodes = set(pattern.output_nodes)
    non_outputs = nodes - output_nodes
    meas_planes = pattern.get_meas_plane()
    meas_angles = pattern.get_angles()
    nodes = set(nodes)

    l_x, l_y, l_z = get_pauli_nodes(meas_planes, meas_angles)

    p_all, l_k = find_pauliflow(g, input_nodes, output_nodes, meas_planes, meas_angles, mode="all")
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
                if ignored_nodes & (l_x | l_y) != ignored_nodes:
                    continue
                odd_neighbers = find_odd_neighbor(g, p_i)
                if zflow_node & odd_neighbers == zflow_node:
                    ignored_nodes = zflow_node - odd_neighbers - {node}
                    # check if nodes in ignored_nodes are measured in Z or Y basis
                    if ignored_nodes & (l_y | l_z) == ignored_nodes:
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
    xflow = dict()
    zflow = dict()
    for cmd in pattern:
        if cmd.kind == CommandKind.M:
            target = cmd.node
            xflow_source = cmd.s_domain & nodes
            zflow_source = cmd.t_domain & nodes
            for node in xflow_source:
                if node not in xflow.keys():
                    xflow[node] = set()
                xflow[node] |= {target}
            for node in zflow_source:
                if node not in zflow.keys():
                    zflow[node] = set()
                zflow[node] |= {target}
        if cmd.kind == CommandKind.X:
            target = cmd.node
            xflow_source = cmd.domain & nodes
            for node in xflow_source:
                if node not in xflow.keys():
                    xflow[node] = set()
                xflow[node] |= {target}
        if cmd.kind == CommandKind.Z:
            target = cmd.node
            zflow_source = cmd.domain & nodes
            for node in zflow_source:
                if node not in zflow.keys():
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
            nb = nb | {edge[1]}
        elif node == edge[1]:
            nb = nb | {edge[0]}
    return nb


def get_min_depth(l_k: dict[int, int]) -> int:
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


def find_odd_neighbor(graph: nx.Graph, vertices: set[int]) -> set[int]:
    """Return the set containing the odd neighbor of a set of vertices.

    Parameters
    ----------
    graph : nx.Graph
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
        dependence_flow = {u: set() for u in inputs}
    except Exception:
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
    combined_flow = dict()
    for node, corrections in flow.items():
        combined_flow[node] = (corrections - (l_x | l_y)) | (odd_flow[node] - (l_y | l_z))
        for ynode in l_y:
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
    layers = dict()
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
    """Get adjacency matrix of the graph.

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
    iset: set[int],
    oset: set[int],
    flow: dict[int, set],
    meas_planes: dict[int, Plane] | None = None,
) -> bool:
    """Check whether the flow is valid.

    Parameters
    ----------
    graph: nx.Graph
        graph (incl. in and out)
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
            valid_flow = False
            return valid_flow

    odd_flow = {node: find_odd_neighbor(graph, corrections) for node, corrections in flow.items()}

    try:
        _, _ = get_layers_from_flow(flow, odd_flow, iset, oset)
    except ValueError:
        valid_flow = False
        return valid_flow
    # check if v ~ f(v) for each node
    edges = set(graph.edges)
    for node, corrections in flow.items():
        if len(corrections) > 1:
            valid_flow = False
            return valid_flow
        correction = next(iter(corrections))
        if (node, correction) not in edges and (correction, node) not in edges:
            valid_flow = False
            return valid_flow
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
    graph: nx.Graph
        graph (incl. in and out)
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
    odd_flow = dict()
    for non_output in non_outputs:
        if non_output not in gflow:
            gflow[non_output] = set()
            odd_flow[non_output] = set()
        else:
            odd_flow[non_output] = find_odd_neighbor(graph, gflow[non_output])

    try:
        _, _ = get_layers_from_flow(gflow, odd_flow, iset, oset)
    except ValueError:
        valid_flow = False
        return valid_flow

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
    graph: nx.Graph
        graph (incl. in and out)
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
    odd_flow = dict()
    for non_output in non_outputs:
        if non_output not in pauliflow.keys():
            pauliflow[non_output] = set()
            odd_flow[non_output] = set()
        else:
            odd_flow[non_output] = find_odd_neighbor(graph, pauliflow[non_output])

    try:
        layers, depth = get_layers_from_flow(pauliflow, odd_flow, iset, oset, (l_x, l_y, l_z))
    except ValueError:
        valid_flow = False
        return valid_flow
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
    check_meas_planes(meas_planes)
    l_x, l_y, l_z = set(), set(), set()
    for node, plane in meas_planes.items():
        if plane == Plane.XY:
            if utils.is_integer(meas_angles[node]):  # measurement angle is integer
                l_x |= {node}
            elif utils.is_integer(2 * meas_angles[node]):  # measurement angle is half integer
                l_y |= {node}
        elif plane == Plane.XZ:
            if utils.is_integer(meas_angles[node]):
                l_z |= {node}
            elif utils.is_integer(2 * meas_angles[node]):
                l_x |= {node}
        elif plane == Plane.YZ:
            if utils.is_integer(meas_angles[node]):
                l_y |= {node}
            elif utils.is_integer(2 * meas_angles[node]):
                l_z |= {node}
    return l_x, l_y, l_z
