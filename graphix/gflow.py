"""Optimum generalized flow finding algorithm

For a given open graph (G, I, O), this iterative method
finds a generalized flow (gflow) [NJP 9, 250 (2007)] in polynomial time.
In particular, this outputs gflow with minimum depth.
We implemented a modification according to definition of
Pauli flow (NJP 9, 250 (2007)).

Ref: Mhalla and Perdrix, International Colloquium on Automata,
Languages, and Programming (Springer, 2008), pp. 857-868.

"""

import networkx as nx
import numpy as np
import z3
import copy


def solvebool(A, b):
    """solves linear equations of booleans

    Solves Ax=b, where A is n*m matrix and b is 1*m array, both of booleans.
    for example, for A=[[0,1,1],[1,0,1]] and b=[1,0], we solve:
        XOR(x1,x2) = 1

        XOR(x0,x2) = 0
    for which (one of) the solution is [x0,x1,x2]=[1,0,1]

    Uses Z3, a theorem prover from Microsoft Research.

    Parameters
    ----------
    A: np.array
        n*m array of 1s and 0s, or booleans
    b: np.array
        length m array of 1s and 0s, or booleans

    Returns
    --------
    x: np.array
        length n array of 1s and 0s satisfying Ax=b.
        if no solution is found, returns False.
    """

    def xor_n(a):
        if len(a) == 1:
            return a[0]
        elif len(a) > 1:
            return z3.Xor(a[0], xor_n(a[1:]))

    # create list of params
    p = []
    for i in range(A.shape[1]):
        p.append(z3.Bool(i))
    p = np.array(p)

    # add constraints
    s = z3.Solver()
    for i in range(len(b)):
        arr = np.asarray(A[i, :]).flatten().astype(np.bool8)
        if arr.any():
            if b[i] == 0:
                s.add(z3.Not(xor_n(p[arr])))
            else:
                s.add(xor_n(p[arr]))

    # solve
    if s.check() == z3.sat:  # there's solution
        ans = s.model()
        return np.array([ans[p[i]] for i in range(A.shape[1])])
    else:  # no solution found
        return False


def gflow(g, v_in, v_out, meas_plane=None, timeout=1000):
    """Optimum generalized flow finding algorithm

    For open graph g with input and output, this returns optimum gflow.

    gflow consist of function g(i) where i is the qubit labels,
    and strict partial ordering < or layers labels l_k where each element
    specify the order of qubits to be measured to maintain determinism in MBQC.
    In practice, we must measure qubits in order specified in array l_k (increasing order
    of l_k from 1), and for each measurements of qubit i we must perform corrections on
    qubits in g(i), depending on the measurement outcome.

    For more details of gflow, see Browne et al., NJP 9, 250 (2007).

    Original algorithm by Mhalla and Perdrix,
    International Colloquium on Automata, Languages, and Programming (Springer, 2008),
    pp. 857-868.

    Parameters
    ----------
    g: nx.Graph
        graph (incl. in and out)
    v_in: set
        set of node labels for input
    v_out: set
        set of node labels for output
    timeout: int
        number of iterations allowed before timeout

    Returns
    -------
    g: list of sets
        list of length |g| where each set is the correcting nodes for
        the measurements of each qubits. function g() in gflow.
    l_k: np.array
        1D array of length |g|, where elements are layer of each qubits
        corresponds to the strict partial ordering < in gflow.
        Measurements must proceed in decreasing order of layer numbers.
    """
    v = set(g.nodes)
    if meas_plane is None:
        meas_plane = {u: "XY" for u in v}

    gamma = nx.to_numpy_matrix(g)  # adjacency matrix
    index_list = list(g.nodes)  # match g.nodes with gamma's index
    l_k = {i: 0 for i in v}  # contains k, layer number initialized to default (0).
    g = dict()  # contains the correction set g(i) for i\in V.
    k = 1
    vo = copy.deepcopy(v_out)
    finished = False
    count = 0
    while not finished:
        count += 1
        vo, g, l_k, finished, exist = gflowaux(v, gamma, index_list, v_in, vo, g, l_k, k, meas_plane)
        k = k + 1
        if not exist:
            return None, None
        if count > timeout:
            raise TimeoutError("max iteration number n={} reached".format(timeout))
    return g, l_k


def gflowaux(v, gamma, index_list, v_in, v_out, g, l_k, k, meas_plane):
    """Function to find one layer of the gflow.

    Ref: Mhalla and Perdrix, International Colloquium on Automata,
    Languages, and Programming (Springer, 2008), pp. 857-868.

    Parameters
    ----------
    v: set
        labels of all qubits (nodes)
    gamma: np.array
        adjacency matrix of graph
    index_list: list of ints
        this list connects between index of gamma and node number of Graph.
    v_in: set
        input qubit set
    v_out: set
        output qubit set U set of qubits in layers 0...k-1.
    g: list of sets
        g(i) for all qubits i
    l_k: np.array
        1D array for all qubits labeling layer number
    k: current layer number.
    meas_plane: array of length |v| containing 'X','Y','Z','XY','YZ','XZ'.
        measurement planes xy, yz, xz or Pauli measurement x,y,z.

    Outputs
    -------
    v_out: set
        output qubit set U set of qubits in layers 0...k+1.
    g: list of sets
        updated g(i) for all qubits i
    l_k: np.array
        updated 1D array for all qubits labeling layer number
    finished: bool
        whether iteration ends or not
    exist: bool
        whether gflow exists or not
    """

    c_set = set()
    v_rem = v - v_out  # remaining vertices
    v_rem_list = list(v_rem)
    n = len(v_rem)
    v_correct = v_out - v_in
    v_correct_list = list(v_correct)

    index_0 = [[index_list.index(i)] for i in iter(v_rem)]  # for slicing rows
    index_1 = [index_list.index(i) for i in iter(v_correct)]  # for slicing columns

    # if index_0 or index_1 is blank, skip the calculation below
    if len(index_0) * len(index_1):
        gamma_sub = gamma[index_0, index_1]

        for u in iter(v_rem):
            if meas_plane[u] == "Z":
                c_set = c_set | {u}
                g[u] = set()
                l_k[u] = k
                continue
            elif meas_plane[u] in ["XY", "XZ", "X", "Y"]:
                Iu = np.zeros(n, dtype=np.int8)
                Iu[v_rem_list.index(u)] = 1
            elif meas_plane[u] == "YZ":
                Iu = np.ones(n, dtype=np.int8)
                Iu[v_rem_list.index(u)] = 0

            Ix = solvebool(gamma_sub.astype(np.int8), Iu.astype(np.int8))
            inds = np.where(Ix)[0]

            if len(inds) > 0:  # has solution
                c_set = c_set | {u}
                g[u] = set(v_correct_list[ind_] for ind_ in inds)
                l_k[u] = k
    if not c_set:
        finished = True
        if v_out == v:
            exist = True
        else:
            exist = False
    else:
        finished = False
        exist = True

    return v_out | c_set, g, l_k, finished, exist


def flow(g, v_in, v_out, meas_plane=None, timeout=10000):
    """Causal flow finding algorithm

    For open graph g with input and output, this returns causal flow.

    For more detail of causal flow, see Danos and Kashefi, PRA 74, 052310 (2006).

    Original algorithm by Mhalla and Perdrix,
    International Colloquium on Automata, Languages, and Programming (2008),
    pp. 857-868.

    Parameters
    ----------
    g: nx.Graph
        graph (incl. in and out)
    v_in: set
        set of node labels for input
    v_out: set
        set of node labels for output
    timeout: int
        number of iterations allowed before timeout

    Returns
    -------
    f: list of nodes
        list of length |g| where each node corrects the measurements of each qubits. function f() in flow.
    l_k: np.array
        1D array of length |g|, where elements are layer of each qubits
        corresponds to the strict partial ordering < in flow.
        Measurements must proceed in decreasing order of layer numbers.
    """
    v = set(g.nodes)
    e = set(g.edges)

    l_k = {i: 0 for i in v}
    f = dict()
    k = 1
    vo = copy.deepcopy(v_out)
    finished = False
    exist = True
    count = 0
    v_c = v_out - v_in
    while not finished:
        count += 1
        vo, v_c, f, l_k, finished, exist = flowaux(v, e, v_in, vo, v_c, f, l_k, k)
        k += 1
        if not exist:
            return None, None
        if count > timeout:
            raise TimeoutError("max iteration number n={} reached".format(timeout))
    return f, l_k


def flowaux(v, e, v_in, v_out, v_c, f, l_k, k):
    """Function to find one layer of the flow.

    Ref: Mhalla and Perdrix, International Colloquium on Automata,
    Languages, and Programming (Springer, 2008), pp. 857-868.

    Parameters
    ----------
    v: set
        labels of all qubits (nodes)
    e: set
        edges
    v_in: set
        input qubit set
    v_out: set
        output qubit set U set of qubits in layers 0...k-1.
    v_c: set
        correction qubit set in layer k
    f: list of sets
        f(i) for all qubits i
    l_k: np.array
        1D array for all qubits labeling layer number
    k: current layer number.
    meas_plane: array of length |v| containing 'x','y','z','xy','yz',xz'.
        measurement planes xy, yz, xz or Pauli measurement x,y,z.

    Outputs
    -------
    v_out: set
        output qubit set U set of qubits in layers 0...k+1.
    v_c: set
        correction qubit set updated in the k-th layer.
    f: list of sets
        updated f(i) for all qubits i
    l_k: np.array
        updated 1D array for all qubits labeling layer number
    finished: bool
        whether iteration ends or not
    exist: bool
        whether gflow exists or not
    """
    v_out_prime = set()
    c_prime = set()

    for q in v_c:
        N = search_neighbor(q, e)
        p_set = N & (v - v_out)
        if len(p_set) == 1:
            p = list(p_set)[0]
            f[p] = q
            l_k[p] = k
            v_out_prime = v_out_prime | {p}
            c_prime = c_prime | {q}
    # determine whether there exists flow
    if not v_out_prime:
        finished = True
        if v_out == v:
            exist = True
        else:
            exist = False
    else:
        finished = False
        exist = True
    return (
        v_out | v_out_prime,
        (v_c - c_prime) | (v_out_prime & (v - v_in)),
        f,
        l_k,
        finished,
        exist,
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


def find_flow(g, v_in, v_out, meas_plane=None, timeout=100):
    """Function to determine whether there exists flow or gflow

    Parameters
    ---------
    g: nx.Graph
        graph (incl. in and out)
    v_in: set
        set of node labels for input
    v_out: set
        set of node labels for output
    timeout: int
        number of iterations allowed before timeout
    """
    f, l_k = gflow(g, v_in, v_out, meas_plane, timeout)
    if f:
        print("gflow found")
        print("g is ", f)
        print("l_k is ", l_k)
    else:
        print("no gflow found, finding flow")
    f, l_k = flow(g, v_in, v_out, timeout=timeout)
    if f:
        print("flow found")
        print("f is ", f)
        print("l_k is ", l_k)
    else:
        print("no flow found")


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
        if np.mod(len(set(graph.neighbor(c)) ^ vertices), 2) == 1:
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
