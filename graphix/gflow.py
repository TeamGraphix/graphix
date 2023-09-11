"""flow finding algorithm

For a given underlying graph (G, I, O, meas_plane), this method finds a (generalized) flow [NJP 9, 250 (2007)] in polynomincal time.
In particular, this outputs gflow with minimum depth, maximally delayed gflow.

Ref: Mhalla and Perdrix, International Colloquium on Automata,
Languages, and Programming (Springer, 2008), pp. 857-868.
Ref: Backens et al., Quantum 5, 421 (2021).

"""

from itertools import product

import networkx as nx
import numpy as np


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

    We use the extended gflow finding algorithm proposed by Backens et al., Quantum 5, 421 (2021).

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
    mode: str
        "single", "all", or "abstract". "single" returns a single solution for each qubit, while "all" returns all solutions. "abstract" returns the abstract solution. user has to substitute uncertainty map into abstract solution manually.

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
    mode: str
        "single", "all", or "abstract". "single" returns a single solution for each qubit, while "all" returns all solutions. "abstract" returns the abstract solution. user has to substitute uncertainty map into abstract solution manually.

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
    solver = GF2Solver()
    adjacency_matrix, node_order_list = get_adjacency_matrix(graph)
    adjacency_matrix_row_reduced = remove_nodes_from_row(adjacency_matrix, node_order_list, output)
    adjacency_matrix_reduced = remove_nodes_from_column(
        adjacency_matrix_row_reduced, node_order_list, nodes - correction_candidate
    )

    node_order_row = []
    node_order_col = []
    for node in node_order_list:
        if node in non_output:
            node_order_row.append(node)
        if node in correction_candidate:
            node_order_col.append(node)

    RHS_map = dict()
    for node in non_output:
        vec = np.zeros(len(node_order_row))
        if meas_planes[node] == "XY":
            vec[node_order_row.index(node)] = 1
        elif meas_planes[node] == "XZ":
            vec[node_order_row.index(node)] = 1
            vec_add = adjacency_matrix_row_reduced[:, node_order_row.index(node)].reshape(vec.shape)
            vec = (vec + vec_add) % 2
        elif meas_planes[node] == "YZ":
            vec = adjacency_matrix_row_reduced[:, node_order_row.index(node)].reshape(vec.shape)
        RHS_map[node] = vec

    all_solutions, uncertainty = solver.solve(adjacency_matrix_reduced, RHS_map, node_order_col)

    corrected = set()
    for candidate in all_solutions.keys():
        if len(all_solutions[candidate]) == 0:
            continue
        if meas_planes[candidate] in ["XZ", "YZ"]:
            all_solutions[candidate][candidate] = SolutionNode(candidate, 1, set())
        if mode == "single":
            g[candidate] = find_single_solution(all_solutions[candidate], uncertainty)
            l_k[candidate] = k
        elif mode == "all":
            g[candidate] = find_all_solutions(all_solutions[candidate], uncertainty)
            l_k[candidate] = k
        elif mode == "abstract":
            g[candidate] = all_solutions[candidate]
            l_k[candidate] = {"layer": k, "uncertainty": uncertainty}

        corrected |= {candidate}

    if len(corrected) == 0:
        if output == nodes:
            return g, l_k
        else:
            return None, None
    else:
        return gflowaux(graph, input, output | corrected, meas_planes, k + 1, l_k, g, mode=mode)


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
        measurement planes for each qubits. meas_planes[i] is the measurement plane for qubit i. Note that an underlying graph has a causal flow only if all measurement planes are "XY".

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


class GF2Solver:
    """Solver for GF(2) linear equations"""

    def __init__(self):
        """Constructor for GF2Solver"""
        pass

    def solve(self, adjacency_matrix, RHS_map, node_order_col):
        """Solve the linear equations

        Parameters
        ----------
        adjacency_matrix: np.array
            reduced adjacency matrix of the graph
        RHS_map: dict
            RHS of the linear equations. RHS_map[NODE] is the right hand side of the equations for the NODE.
        node_order_col: list
            ordered list of nodes. node_order_col[i] is the node label of i-th column of the adjacency matrix.

        Returns
        -------
        all_solutions: dict
            solutions of the linear equations. all_solutions[NODE] is the gflow map of NODE. Note that solutions generally contain uncertainty because the equations are not always square. Therefore, solutions in this step is represented by abstract class. See `SolutionNode` and `find_all_solutions` for details.
        uncertainty: set
            set of nodes which are not determined by the linear equations.
        """
        LHS, RHS, node_order_col, RHS_order = self.forward_elimination(adjacency_matrix, node_order_col, RHS_map)
        all_solutions, uncertainty = self.backward_substitution(LHS, RHS, node_order_col, RHS_order)
        return all_solutions, uncertainty

    @staticmethod
    def forward_elimination(adjacency_matrix, node_order_col, RHS_map):
        """Forward elimination of the linear equations

        Parameters
        ----------
        adjacency_matrix: np.array
            reduced adjacency matrix of the graph
        node_order_col: list
            ordered list of nodes. node_order_col[i] is the node label of i-th column of the adjacency matrix.
        RHS_map: dict
            RHS of the linear equations. RHS_map[NODE] is the right hand side of the equations for the NODE.

        Returns
        -------
        LHS: np.array
            LHS of the linear equations. Generally, LHS is not a square matrix.
        RHS: np.array
            RHS of the linear equations.
        node_order_col: list
            ordered list of nodes. node_order_col[i] is the node label of i-th column of the adjacency matrix. This is sometimes modified when the pivot column is swapped.
        RHS_order: list
            ordered list of nodes. RHS_order[i] is the node label of i-th column of the RHS matrix.
        """
        dim_O_notI = adjacency_matrix.shape[1]

        RHS_order = [node for node in RHS_map.keys()]
        B = np.zeros((len(adjacency_matrix), len(RHS_map)))
        for node, vec in RHS_map.items():
            ind = RHS_order.index(node)
            B[:, ind] = vec

        expanded_matrix = np.concatenate((adjacency_matrix, B), axis=1)

        # gaussian elimination
        max_rank = min(adjacency_matrix.shape)
        for row in range(max_rank):
            # find the pivot row
            if expanded_matrix[row, row] == 0:
                pivot_row = np.where(expanded_matrix[row:, row] != 0)[0]
                if len(pivot_row) == 0:
                    # pivot column
                    pivot_col = np.where(expanded_matrix[row:, row : adjacency_matrix.shape[1]] != 0)
                    if len(pivot_col[1]) == 0:
                        break
                    pivot_col = pivot_col[1][0] + row
                    # swap the pivot column with the current column
                    expanded_matrix[:, [row, pivot_col]] = expanded_matrix[:, [pivot_col, row]]
                    # swap the indices vector
                    node_order_col[row], node_order_col[pivot_col] = (
                        node_order_col[pivot_col],
                        node_order_col[row],
                    )
                    # find the pivot row
                    pivot_row = np.where(expanded_matrix[row:, row] != 0)[0]
                    assert len(pivot_row) > 0, "pivot row not found"

                pivot_row = pivot_row[0] + row
                # swap the pivot row with the current row
                expanded_matrix[[row, pivot_row]] = expanded_matrix[[pivot_row, row]]
            # perform row operations to eliminate the current column
            for row_eliminated in range(row + 1, expanded_matrix.shape[0]):
                if expanded_matrix[row_eliminated, row] != 0:
                    expanded_matrix[row_eliminated, :] = (expanded_matrix[row_eliminated] + expanded_matrix[row, :]) % 2

        LHS = expanded_matrix[:, :dim_O_notI]
        RHS = expanded_matrix[:, dim_O_notI:]
        return LHS, RHS, node_order_col, RHS_order

    @staticmethod
    def backward_substitution(LHS, RHS, node_order_col, RHS_order):
        """Backward substitution of the linear equations

        Parameters
        ----------
        LHS: np.array
            LHS of the linear equations. Generally, LHS is not a square matrix.
        RHS: np.array
            RHS of the linear equations.
        node_order_col: list
            ordered list of nodes. node_order_col[i] is the node label of i-th column of the adjacency matrix.
        RHS_order: list
            ordered list of nodes. RHS_order[i] is the node label of i-th column of the RHS matrix.

        Returns
        -------
        all_solutions: dict
            solutions of the linear equations. all_solutions[NODE] is the gflow map of NODE. Note that solutions generally contain uncertainty because the equations are not always square. Therefore, solutions in this step is represented by abstract class. See `SolutionNode` and `find_all_solutions` for details.
        uncertainty: set
            set of nodes which are not determined by the linear equations.
        """
        # find the rank of the LHS matrix
        all_solutions = dict()

        for j in range(len(RHS_order)):
            candidate = RHS_order[j]
            RHS_vector = RHS[:, j]
            # perform back substitution
            all_solutions[candidate] = dict()
            rank = min(LHS.shape)
            for row in range(LHS.shape[0] - 1, -1, -1):
                if row >= rank:
                    if RHS_vector[row] != 0:
                        # no solution. empty dict is returned
                        break
                    continue

                if LHS[row, row] == 0:
                    if RHS_vector[row] != 0:
                        # no solution. empty dict is returned
                        break
                    else:
                        rank -= 1
                        continue
                # find the current solution
                uncertainty = set()
                for col in range(rank, LHS.shape[1]):
                    if LHS[row, col] != 0:
                        uncertainty |= {node_order_col[col]}
                currect_sol = SolutionNode(node_order_col[row], RHS_vector[row, 0], uncertainty)
                for col in range(row + 1, rank):
                    if LHS[row, col] != 0:
                        currect_sol.subtract(all_solutions[candidate][node_order_col[col]])
                all_solutions[candidate][node_order_col[row]] = currect_sol
        uncertainty = {node_order_col[col] for col in range(rank, LHS.shape[1])}

        return all_solutions, uncertainty


def find_single_solution(solution, uncertainty):
    """
    Find a single gflow solution. This is a special case of `find_all_solutions()`. See `find_all_solutions()` for details.

    Parameters
    ----------
    solution: dict
        solution for specified node.
    uncertainty: set
        set of nodes which are not determined by the linear equations.

    Returns
    -------
    substituted_solution: set
        set of nodes which are determined by the uncertainty map.
    """
    substituted_solution = set()
    uncertainty_map = dict(zip(uncertainty, [0] * len(uncertainty)))
    for solution_node in solution.values():
        if solution_node.substitute(uncertainty_map):
            substituted_solution |= {solution_node.node_index}
    return substituted_solution


def find_all_solutions(solutions, uncertainty):
    """
    Find all gflow solutions.

    Parameters
    ----------
    solutions: dict
        solution for specified node.
    uncertainty: set
        set of nodes which are not determined by the linear equations.

    Returns
    -------
    all_substituded_solutions: dict
        all possible maximally delayed gflow solutions. Keys of the dict are the uncertainty map. Values of the dict are the set of nodes which are determined by the uncertainty map.
    """
    if uncertainty == set():
        all_substituded_solutions = dict()
        all_substituded_solutions["no uncertainty"] = set()
        for solution_node in solutions.values():
            if solution_node.substitute({}):
                all_substituded_solutions["no uncertainty"] |= {solution_node.node_index}
        return all_substituded_solutions
    all_uncertainty = product([0, 1], repeat=len(uncertainty))
    all_substituded_solutions = dict()
    for uncertainty_values in all_uncertainty:
        uncertainty_map = dict(zip(uncertainty, uncertainty_values))
        key = str(uncertainty_map)
        all_substituded_solutions[key] = set()
        for solution_node in solutions.values():
            if solution_node.substitute(uncertainty_map):
                all_substituded_solutions[key] |= {solution_node.node_index}
            for node, value in uncertainty_map.items():
                if value:
                    all_substituded_solutions[key] |= {node}
    return all_substituded_solutions


def get_adjacency_matrix(graph):
    """Get adjacency matrix of the graph

    Returns
    -------
    adjacency_matrix: np.array
        adjacency matrix of the graph
    node_list: list
        ordered list of nodes. node_list[i] is the node label of i-th row/column of the adjacency matrix.

    """
    node_list = list(graph.nodes)
    node_list.sort()
    adjacency_matrix = nx.adjacency_matrix(graph, nodelist=node_list).todense()
    return adjacency_matrix, node_list


def remove_nodes_from_row(matrix, node_order_list, nodes):
    """
    Remove the selected rows from the adjacency matrix.

    Parameters
    ----------
    matrix: np.array
        adjacency matrix
    node_order_list: list
        ordered list of nodes. node_order_list[i] is the node label of i-th row/column of the adjacency matrix.
    nodes: list
        list of nodes to be removed.

    Returns
    -------
    matrix: np.array
        adjacency matrix with removed rows.
    """
    rows = [node_order_list.index(node) for node in nodes]
    # create a copy of the matrix
    matrix = matrix.copy()
    # remove the selected rows
    matrix = np.delete(matrix, rows, axis=0)
    # return the matrix
    return matrix


def remove_nodes_from_column(matrix, node_order_list, nodes):
    """
    Remove the selected columns from the adjacency matrix.

    Parameters
    ----------
    matrix: np.array
        adjacency matrix
    node_order_list: list
        ordered list of nodes. node_order_list[i] is the node label of i-th row/column of the adjacency matrix.

    Returns
    -------
    matrix: np.array
        adjacency matrix with removed columns.
    """
    columns = [node_order_list.index(node) for node in nodes]
    # create a copy of the matrix
    matrix = matrix.copy()
    # remove the selected columns
    matrix = np.delete(matrix, columns, axis=1)
    # return the matrix
    return matrix


class SolutionNode:
    """
    Class to represent a solution node. This is an abstract class for uncertainty of gflow.
    """

    def __init__(self, node_index, parity, uncertainty):
        """
        Constructor for SolutionNode

        Parameters
        ----------
        node_index: int
            node index of the solution node.
        parity: int
            parity of the solution node.
        uncertainty: set
            set of nodes which are not determined by the linear equations.
        """
        self.node_index = node_index
        self.parity = parity
        self.uncertainty = uncertainty

    def substitute(self, uncertainty_map):
        """
        Substitute the uncertainty map to the solution node.

        Parameters
        ----------
        uncertainty_map: dict
            uncertainty map. uncertainty_map[node] is the parity of the node.

        Returns
        -------
        solution: bool
            whether the solution node is included in the correction map or not
        """
        solution = self.parity
        for undeterminded_node in self.uncertainty:
            solution += uncertainty_map[undeterminded_node]
        solution = solution % 2
        return solution

    def subtract(self, other):
        """
        Subtract the other solution node from the current solution node.

        Parameters
        ----------
        other: SolutionNode
            other solution node to be subtracted.
        """
        self.parity = (self.parity + other.parity) % 2
        self.uncertainty - other.uncertainty

    def print_solution(self):
        """
        Print the solution node.
        """
        print("node index: ", self.node_index)
        print("parity: ", self.parity)
        print("uncertainty nodes: ", self.uncertainty)
