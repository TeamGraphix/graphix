"""flow finding algorithm

For a given underlying graph (G, I, O, meas_plane), this method finds a (generalized) flow [NJP 9, 250 (2007)]
in polynomincal time.
In particular, this outputs gflow with minimum depth, maximally delayed gflow.

Ref: Mhalla and Perdrix, International Colloquium on Automata,
Languages, and Programming (Springer, 2008), pp. 857-868.
Ref: Backens et al., Quantum 5, 421 (2021).

"""

from itertools import product

import networkx as nx
import numpy as np
import sympy as sp

from graphix.linalg import MatGF2


def gflow(graph, input, output, meas_planes, mode="single"):
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
    graph,
    input: set,
    output: set,
    meas_planes: dict,
    k: int,
    l_k: dict,
    g: dict,
    mode,
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
            vec.data = adj_mat_row_reduced.data[:, i_row].reshape(vec.data.shape)
        b.data[:, i_row] = vec.data

    adj_mat, b, _, col_pertumutation = adj_mat.forward_eliminate(b)
    x, kernels = adj_mat.backward_substitute(b)

    corrected_nodes = set()
    for i_row in range(len(node_order_row)):
        non_out_node = node_order_row[i_row]
        x_col = x[:, i_row]
        if x_col[0] == sp.nan:  # no solution
            continue
        if mode == "single":
            sol_list = [x_col[i].subs(zip(kernels, [sp.false] * len(kernels))) for i in range(len(x_col))]
            sol = np.array(sol_list)
            sol_index = sol.nonzero()[0]
            g[non_out_node] = set(node_order_col[col_pertumutation[i]] for i in sol_index)
            if meas_planes[non_out_node] in ["XZ", "YZ"]:
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
                g_i = set(node_order_col[col_pertumutation[i]] for i in sol_index)
                if meas_planes[non_out_node] in ["XZ", "YZ"]:
                    g_i |= {non_out_node}

                g[non_out_node] |= {frozenset(g_i)}

        elif mode == "abstract":
            g[non_out_node] = dict()
            for i in range(len(x_col)):
                node = node_order_col[col_pertumutation[i]]
                g[non_out_node][node] = x_col[i]
            for i in range(len(kernels)):
                g[non_out_node][node_order_col[col_pertumutation[len(x_col) + i]]] = kernels[i]
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


def flow(graph, input, output, meas_planes=None):
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


def flowaux(nodes, edges, input, output, v_c, f, l_k, k):
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
            f[p] = q
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


def pauliflow(graph, input, output, meas_planes=None, meas_angles=None):
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

    Returns
    -------
    p: dict
        Pauli flow function. p[i] is the set of qubits to be corrected for the measurement of qubit i.
    l_k: dict
        layers obtained by  Pauli flow algorithm. l_k[d] is a node set of depth d.
    """
    l_k = dict()
    p = dict()
    Lx, Ly, Lz = set(), set(), set()
    for node in graph.nodes:
        if node in output:
            l_k[node] = 0
        elif meas_planes[node] == "XY":
            if meas_angles[node] == 0:
                Lx |= {node}
            elif meas_angles[node] == 1 / 2:
                Ly |= {node}
        elif meas_planes[node] == "ZX":
            if meas_angles[node] == 0:
                Lz |= {node}
            elif meas_angles[node] == 1 / 2:
                Lx |= {node}
        elif meas_planes[node] == "YZ":
            if meas_angles[node] == 0:
                Ly |= {node}
            elif meas_angles[node] == 1 / 2:
                Lz |= {node}
    print("Lx is ", Lx)
    print("Ly is ", Ly)
    print("Lz is ", Lz)
    print("l_k is ", l_k)
    return pauliflowaux(graph, input, output, meas_planes, 0, set(), output, l_k, p, Lx, Ly, Lz)


def pauliflowaux(
    graph,
    input: set,
    output: set,
    meas_planes: dict,
    k: int,
    correction_candidate: set,
    solved_nodes: set,
    l_k: dict,
    p: dict,
    Lx: set,
    Ly: set,
    Lz: set,
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
    Lx: set
        set of qubits whose measurement operator is "X".
    Ly: set
        set of qubits whose measurement operator is "Y".
    Lz: set
        set of qubits whose measurement operator is "Z".

    Returns
    -------
    p: dict
        Pauli flow function. p[i] is the set of qubits to be corrected for the measurement of qubit i.
    l_k: dict
        layers obtained by Pauli flow algorithm. l_k[d] is a node set of depth d.
    """
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
    print("node_order_row is \n", node_order_row)
    print("node_order_col is \n", node_order_col)
    print("adj_mat is \n", adj_mat)
    print("node_order_row_lower is \n", node_order_row_lower)
    print("adj_mat_w_id is \n", adj_mat_w_id)

    Pbar = correction_candidate | Ly | Lz
    P = nodes - Pbar
    K = (correction_candidate | Lx | Ly) & (nodes - input)
    Y = Ly - correction_candidate
    print("P is \n", P)
    print("K is \n", K)
    print("Y is \n", Y)

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
        print("\n -Node is ", node)
        print("node_order_row_ is \n", node_order_row_)
        print("node_order_col_ is \n", node_order_col_)
        print("adj_mat_ is \n", adj_mat_)
        print("node_order_row_lower_ is \n", node_order_row_lower_)
        print("adj_mat_w_id_ is \n", adj_mat_w_id_)
        adj_mat_.concatenate(adj_mat_w_id_, axis=0)

        solved = False
        if not solved and (meas_planes[node] == "XY" or node in Lx or node in Ly):
            S = MatGF2(np.zeros((len(node_order_row_), 1), dtype=int))
            S.data[node_order_row_.index(node), :] = 1
            print("S_upper is \n", S)
            S_lower = MatGF2(np.zeros((len(node_order_row_lower_), 1), dtype=int))
            print("S_lower is \n", S_lower)
            S.concatenate(S_lower, axis=0)
            print("S is \n", S)
            adj_mat_XY, S, _, col_pertumutation_XY = adj_mat_.forward_eliminate(S, copy=True)
            x_XY, kernels_XY = adj_mat_XY.backward_substitute(S)
            if x_XY[0, 0] != sp.nan:
                print("x_XY is \n", x_XY)
                solved = True
                solved_update |= {node}
                p[node] = set()
                for i in range(x_XY.shape[0]):
                    if x_XY[i, 0]:
                        p[node] |= {node_order_col_[col_pertumutation_XY[i]]}
                l_k[node] = k

        if not solved and (meas_planes[node] == "ZX" or node in Lz or node in Lx):
            S = MatGF2(np.zeros((len(node_order_row_), 1), dtype=int))
            S.data[node_order_row_.index(node)] = 1
            for neighbor in search_neighbor(node, graph.edges):
                if neighbor in P and neighbor in P | {node}:
                    S.data[node_order_row_.index(neighbor), :] = 1
            S_lower = MatGF2(np.zeros((len(node_order_row_lower_), 1), dtype=int))
            for neighbor in search_neighbor(node, graph.edges):
                if neighbor in Y and neighbor in Y - {node}:
                    S_lower.data[node_order_row_lower_.index(neighbor), :] = 1
            S.concatenate(S_lower, axis=0)
            adj_mat_ZX, S, _, col_pertumutation_ZX = adj_mat_.forward_eliminate(S)
            x_ZX, kernels_ZX = adj_mat_ZX.backward_substitute(S)
            if x_ZX[0, 0] != sp.nan:
                solved = True
                solved_update |= {node}
                p[node] = set()
                for i in range(x_ZX.shape[0]):
                    if x_ZX[i, 0]:
                        p[node] |= {node_order_col_[col_pertumutation_ZX[i]]}
                l_k[node] = k

        if not solved and (meas_planes[node] == "YZ" or node in Ly or node in Lz):
            S = MatGF2(np.zeros((len(node_order_row_), 1), dtype=int))
            for neighbor in search_neighbor(node, graph.edges):
                if neighbor in P and neighbor in P | {node}:
                    S.data[node_order_row_.index(neighbor), :] = 1
            S_lower = MatGF2(np.zeros((len(node_order_row_lower_), 1), dtype=int))
            for neighbor in search_neighbor(node, graph.edges):
                if neighbor in Y and neighbor in Y - {node}:
                    S_lower.data[node_order_row_lower_.index(neighbor), :] = 1
            S.concatenate(S_lower, axis=0)
            adj_mat_YZ, S, _, col_pertumutation_YZ = adj_mat_.forward_eliminate(S)
            x_YZ, kernels_YZ = adj_mat_YZ.backward_substitute(S)
            if x_YZ[0, 0] != sp.nan:
                solved = True
                solved_update |= {node}
                p[node] = set()
                for i in range(x_YZ.shape[0]):
                    if x_YZ[i, 0]:
                        p[node] |= {node_order_col_[col_pertumutation_YZ[i]]}
                l_k[node] = k

    if solved_update == set() and k > 0:
        if solved_nodes == nodes:
            return p, l_k
        else:
            return None, None
    else:
        B = solved_nodes | solved_update
        return pauliflowaux(
            graph,
            input,
            output,
            meas_planes,
            k + 1,
            B,
            B,
            l_k,
            p,
            Lx,
            Ly,
            Lz,
        )


def search_neighbor(node, edges):
    """Function to find neighborhood of node in edges. This is an ancillary method for `flowaux()`.

    Parameter
    -------
    node: int
        target node number whose neighboring nodes will be collected
    edges: list of taples
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


def find_flow(graph, input, output, meas_planes=None, mode="single"):
    """Function to determine whether there exists flow or gflow

    Parameters
    ---------
    graph: nx.Graph
        graph (incl. in and out)
    input: set
        set of node labels for input
    output: set
        set of node labels for output
    meas_planes: dict(optional)
        measurement planes for each qubits. meas_planes[i] is the measurement plane for qubit i.
    mode: str(optional)
        This is the option for gflow.
        The gflow finding algorithm can yield multiple equivalent solutions. so there are three options
            - "single": Returrns a single solution
            - "all": Returns all possible solutions
            - "abstract": Returns an abstract solution. Uncertainty is represented with sympy.Symbol objects,
            requiring user substitution to get a concrete answer.
    """
    if meas_planes is None:
        meas_planes = {i: "XY" for i in (set(graph.nodes) - output)}
    f, l_k = flow(graph, input, output, meas_planes)
    if f:
        print("flow found")
        print("f is ", f)
        print("l_k is ", l_k)
    else:
        print("no flow found, finding gflow")
    g, l_k = gflow(graph, input, output, meas_planes, mode=mode)
    if g:
        print("gflow found")
        print("g is ", g)
        print("l_k is ", l_k)
    else:
        print("no gflow found")


def get_min_depth(l_k):
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


def find_odd_neighbor(graph, candidate, vertices):
    """Returns the list containing the odd neighbor of a set of vertices.

    Parameters
    ----------
    graph : networkx.Graph
        underlying graph.
    candidate : iterable
        possible odd neighbors
    vertices : set
        set of nodes indices to find odd neighbors

    Returns
    -------
    out : list
        list of indices for odd neighbor of set `vertices`.
    """
    out = []
    for c in candidate:
        if np.mod(len(set(graph.neighbors(c)) ^ vertices), 2) == 1:
            out.append(c)
    return out


def get_layers(l_k):
    """get components of each layer.
    Parameters
    -------
    l_k: dict
        layers obtained by flow or gflow algorithms

    Returns
    -------
    d: int
        minimum depth of graph
    layers: dict of lists
        components of each layer
    """
    d = get_min_depth(l_k)
    layers = {k: [] for k in range(d + 1)}
    for i in l_k.keys():
        layers[l_k[i]].append(i)
    return d, layers


def get_adjacency_matrix(graph):
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
