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
    """solves linear euqations of booleans

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


def gflow(g, v_in, v_out, meas_plane=None, timeout=100):
    """Optimum generalized flow finding algorithm

    For open graph g with input and output, this returns optimum gflow.

    gflow consist of function g(i) where i is the qubit labels,
    and strict partial ordering < or layers labels l_k where each element
    specify the order of qubits to be measured to maintain determinism in MBQC.
    In practice, we must measure qubits in order specified in array l_k (increasing order
    of l_k from 1), and for each measurements of qubit i we must perform corrections on
    qubits in g(i), depending on the measurement outcome.

    For more details of gflow, see [NJP 9, 250 (2007)].

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
    nqubit = len(g.nodes)
    v = set(np.arange(nqubit))
    if meas_plane is None:
        meas_plane = ['xy' for u in v]

    gamma = nx.to_numpy_matrix(g)  # adjacency matrix
    l_k = np.ones(nqubit, dtype=np.int8) * -1  # contains k, layer number initialized to default (-1).
    g = [set() for i in range(nqubit)]  # contains the correction set g(i) for i\in V.
    k = 1
    vo = copy.deepcopy(v_out)
    finished = False
    count = 0
    while not finished:
        count += 1
        vo, g, l_k = gflowaux(v, gamma, v_in, vo, g, l_k, k, meas_plane)
        k = k + 1
        if len(np.where(l_k == -1)[0]) == len(v_in):  # assigned layer to all qubits except output
            finished = True
        if count > timeout:
            raise TimeoutError('max iteration number n={} reached'.format(timeout))
    return g, l_k


def gflowaux(v, gamma, v_in, v_out, g, l_k, k, meas_plane):
    """Function to find one layer of the gflow.

    Ref: Mhalla and Perdrix, International Colloquium on Automata,
    Languages, and Programming (Springer, 2008), pp. 857-868.

    Parameters
    ----------
    v: set
        labels of all qubits (nodes)
    gamma: np.array
        adjacency matrix of graph
    v_in: set
        input qubit set
    v_out: set
        output qubit set U set of qubits in layers 0...k-1.
    g: list of sets
        g(i) for all qubits i
    l_k: np.array
        1D array for all qubits labeling layer number
    k: current layer number.
    meas_plane: array of length |v| containing 'x','y','z','xy','yz',xz'.
        measurement planes xy, yz, xz or Pauli measurement x,y,z.

    Outputs
    -------
    v_out: set
        output qubit set U set of qubits in layers 0...k+1.
    g: list of sets
        updated g(i) for all qubits i
    l_k: np.array
        updated 1D array for all qubits labeling layer number
    """
    assert list(v) == list(np.arange(len(v)))  # vertex labels must be 0,1,...,len(v)-1.

    c_set = set()
    v_rem = v - v_out  # remaining vertices
    v_rem_list = list(v_rem)
    n = len(v_rem)
    v_correct = v_out - v_in
    v_correct_list = list(v_correct)

    index_0 = [[i] for i in iter(v_rem)]  # for slicing rows
    index_1 = list(v_correct)  # for slicing columns

    gamma_sub = gamma[index_0, index_1]

    for u in iter(v_rem):
        if meas_plane[u] == 'z':
            c_set = c_set | {u}
            g[u] = set()
            l_k[u] = k
            continue
        elif meas_plane[u] in ['xy', 'xz', 'x', 'y']:
            Iu = np.zeros(n, dtype=np.int8)
            Iu[v_rem_list.index(u)] = 1
        elif meas_plane[u] == 'yz':
            Iu = np.ones(n, dtype=np.int8)
            Iu[v_rem_list.index(u)] = 0

        Ix = solvebool(gamma_sub.astype(np.int8), Iu.astype(np.int8))
        inds = np.where(Ix)[0]

        if len(inds) > 0:  # has solution
            c_set = c_set | {u}
            g[u] = set(v_correct_list[ind_] for ind_ in inds)
            l_k[u] = k

    return v_out | c_set, g, l_k
