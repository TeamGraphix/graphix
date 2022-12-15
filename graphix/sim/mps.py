import numpy as np
import tensornetwork as tn
import networkx as nx
from graphix.clifford import CLIFFORD
from graphix.sim.statevec import meas_op
from copy import copy


class MPS:
    """Matrix Product Simulator for MBQC

    Executes the measurement pattern.
    This is a simple implementation. Improved CPU/GPU MPS simulator will be released soon.
    """

    def __init__(self, pattern, singular_value=None, max_truncation_err=None, graph_prep="opt"):
        """

        Parameters
        ----------
        pattern : graphix.Pattern
            MBQC command sequence to be simulated
        singular_value: int
            cut off threshold for SVD decomposition
        truncation_err: float
            cut off threshold for SVD decomposition. truncate maximum number of singular values within truncation_err
        graph_prep : str
            'sequential' for standard method, 'opt' for faster method
        """
        nodes, edges = pattern.get_graph()
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        self.ptn = pattern
        self.results = copy(pattern.results)
        self.graph = G
        # dict of tensornetwork.Node, index should be int
        self.nodes = dict()
        self.state = None
        self.singular_value = singular_value
        self.truncation_err = max_truncation_err
        self.graph_prep = graph_prep
        # accumulated truncation square error
        self.accumulated_err = 0.0

    def set_singular_value(self, chi):
        """Set the number of singular values holding under singular value decomposition(SVD).

        Note: If you choose 'opt' as a graph_prep, you don't have to specify the singular_value.

        Parameters
        ----------
            chi(int): number of singular values holding when SVD executed
        """
        self.singular_value = chi

    def set_truncation_err(self, truncation_err):
        """Set the max truncation error under SVD.

        Note: If you choose 'opt' as a graph_prep, you don't have to specify the truncation_err.

        Parameters
        ----------
            truncation_err (float): Max truncation error allowed under SVD.
            Maximum number of singular values keeping truncation error under specified value will be possessed.
        """
        self.truncation_err = truncation_err

    def count_maxE(self):
        """Count the max number of edges per a node. When maxE is large number, huge memory(2^maxE) will be used.

        Returns
        -------
            int: The max number of edges.
        """
        maxE = 0
        for node in self.graph.nodes:
            n = len(self.graph[node])
            if n > maxE:
                maxE = n
        return maxE

    def add_nodes(self, cmds):
        """add qubits to node sets from command sequence"""
        if self.graph_prep == "sequential":
            for cmd in cmds:
                self.add_node(cmd[1])
        elif self.graph_prep == "opt":
            pass

    def add_node(self, n):
        """Internal method of run_cmd(). Add new qubit to a node set of MPS.

        Parameters
        ----------
            n (int): Site index of the new node.
        """
        neighbor = self.graph.neighbors(n)
        dim = [2]
        axis_names = [str(n)]
        for neighbor_node in neighbor:
            dim.append(1)
            axis_names.append(str(neighbor_node))
        node = tn.Node(np.ones(dim), str(n), axis_names)
        self.nodes[n] = node

    def entangle_nodes(self, edge):
        """Make entanglement between nodes specified by edge. Contract non-dangling edges in this process.
        Optimized contraction will be implemented in a later release.

        Parameters
        ----------
            edge (taple of ints): edge specifies two nodes applied CZ gate.
        """
        if self.graph_prep == "sequential":
            # prepare CZ operator
            cz = tn.Node(
                np.array(
                    [
                        [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]],
                        [[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, -1.0]]],
                    ]
                ),
                name="cz",
                axis_names=["c1", "c2", "c3", "c4"],
            )
            # call nodes from nodes list
            node1 = self.nodes[edge[0]]
            node2 = self.nodes[edge[1]]
            # contract the edge between nodes
            cont_edge = node1[str(edge[1])] ^ node2[str(edge[0])]
            axis_name1 = copy(node1.axis_names)
            axis_name2 = copy(node2.axis_names)
            axis_name1.remove(str(edge[1]))
            axis_name2.remove(str(edge[0]))
            connected = tn.contract(cont_edge, name="connected", axis_names=axis_name1 + axis_name2)
            connected[str(edge[0])] ^ cz["c1"]
            connected[str(edge[1])] ^ cz["c2"]
            connected_rem = copy(connected.edges)
            connected_rem.remove(connected[str(edge[0])])
            connected_rem.remove(connected[str(edge[1])])
            connected_axis_names = copy(connected.axis_names)
            connected_axis_names.remove(str(edge[0]))
            connected_axis_names.remove(str(edge[1]))
            # apply entangle operatorn and rename edges
            entangled = tn.contract_between(
                connected,
                cz,
                name="entangled",
                output_edge_order=[cz["c3"], cz["c4"]] + connected_rem,
                axis_names=[str(edge[0]), str(edge[1])] + connected_axis_names,
            )
            leftedges = []
            for name in axis_name1:
                leftedges.append(entangled[name])
            rightedges = []
            for name in axis_name2:
                rightedges.append(entangled[name])
            # separate 4rank tensor to two 3rank tensors
            node1_new, node2_new, truncation_errs = tn.split_node(
                entangled,
                left_edges=leftedges,
                right_edges=rightedges,
                max_singular_values=self.singular_value,
                max_truncation_err=self.truncation_err,
                left_name=node1.name,
                right_name=node2.name,
                edge_name="contracted",
            )
            # update the nodes
            node1_new.axis_names = axis_name1 + [str(edge[1])]
            node2_new.axis_names = [str(edge[0])] + axis_name2
            self.nodes[edge[0]] = node1_new
            self.nodes[edge[1]] = node2_new
            self.accumulated_err += np.linalg.norm(truncation_errs)
            assert node1_new[node1.name].is_dangling()
            assert node2_new[node2.name].is_dangling()
        elif self.graph_prep == "opt":
            pass

    def initialize(self):
        """initialize the internal MPS state"""
        if self.graph_prep == "sequential":
            self.make_initial()
        elif self.graph_prep == "opt":
            self.make_graph_state()

    def make_initial(self):
        """This is an internal method of run_cmd.
        Prepare a graph state sequentially by applying CZ gates.
        """
        # prepare nodes
        for n in self.graph.nodes:
            self.add_node(n)
        # make entanglement
        for edge in self.graph.edges:
            self.entangle_nodes(edge)

    def finalize(self):
        self.state = self

    def make_graph_state(self):
        """Prepare a graph state with efficient expression, instead of using CZ gates.
        This is is an internal method of run_cmd().

        .. seealso:: :meth:`~graphix.sim.mps.make_initial()`
        """
        # basic vectors
        plus = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)])
        minus = np.array([1.0 / np.sqrt(2), -1.0 / np.sqrt(2)])
        zero = np.array([1.0, 0.0])
        one = np.array([0.0, 1.0])
        for site in self.graph.nodes:
            neighbor = self.graph.neighbors(site)
            A0 = []
            A1 = []
            axis_names = [str(site)]
            edge_order0 = []
            edge_order1 = []
            for n in neighbor:
                if n > site:
                    zero_node_cp = tn.Node(zero)
                    one_node_cp = tn.Node(one)
                    A0.append(zero_node_cp)
                    A1.append(one_node_cp)
                    edge_order0.append(zero_node_cp[0])
                    edge_order1.append(one_node_cp[0])
                elif n < site:
                    plus_node_cp = tn.Node(plus)
                    minus_node_cp = tn.Node(minus)
                    A0.append(plus_node_cp)
                    A1.append(minus_node_cp)
                    edge_order0.append(plus_node_cp[0])
                    edge_order1.append(minus_node_cp[0])
                axis_names.append(str(n))
            if len(A0) * len(A1):
                tensor = np.array(
                    [
                        tn.outer_product_final_nodes(A0, edge_order0).tensor,
                        tn.outer_product_final_nodes(A1, edge_order1).tensor,
                    ]
                )
            else:  # branch for not concatenated graph
                tensor = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)])
            node = tn.Node(tensor, str(site), axis_names=axis_names)
            self.nodes[site] = node

        # connecting all edges
        for edge in self.graph.edges:
            node1 = self.nodes[edge[0]]
            node2 = self.nodes[edge[1]]
            node1[str(edge[1])] ^ node2[str(edge[0])]

    def measure(self, cmd):
        """Perform measurement of a node. In MPS, to apply measurement operator to the tensor,
        consisting Matrix Product State, equals to perform measurement.

        Parameters
        ----------
            cmd (list): measurement command : ['M', node, plane angle, s_domain, t_domain]
        """
        # choose the measurement result randomly
        result = np.random.choice([0, 1])
        self.results[cmd[1]] = result

        # extract signals for adaptive angle
        s_signal = np.sum([self.results[j] for j in cmd[4]])
        t_signal = np.sum([self.results[j] for j in cmd[5]])
        angle = cmd[3] * np.pi * (-1) ** s_signal + np.pi * t_signal
        if len(cmd) == 7:
            m_op = meas_op(angle, vop=cmd[6], plane=cmd[2], choice=result)
        else:
            m_op = meas_op(angle, plane=cmd[2], choice=result)

        # the procedure described below tends to keep the norm of MPS
        buffer = 2**0.5
        m_op = m_op * buffer

        node_op = tn.Node(m_op)
        self.apply_one_site_operator(cmd[1], node_op)

    def correct_byproduct(self, cmd):
        """Perform byproduct correction.

        Parameters
        ----------
            cmd (list): correct for the X or Z byproduct operators, by applying the X or Z gate.
        """
        if np.mod(np.sum([self.results[j] for j in cmd[2]]), 2) == 1:
            if cmd[0] == "X":
                op = np.array([[0.0, 1.0], [1.0, 0.0]])
            elif cmd[0] == "Z":
                op = np.array([[1.0, 0.0], [0.0, -1.0]])
            node_op = tn.Node(op)
            self.apply_one_site_operator(cmd[1], node_op)

    def apply_clifford(self, cmd):
        """Apply single-qubit Clifford gate,
        specified by vop index specified in graphix.clifford.CLIFFORD

        Parameters
        ----------
            cmd (list): clifford command. See {Ref} graphix-paper for the detail. ToDo
        """
        node_op = tn.Node(CLIFFORD[cmd[2]])
        self.apply_one_site_operator(cmd[1], node_op)

    def apply_one_site_operator(self, loc, node_op):
        """Internal method for measure, correct_byproduct, and apply_clifford. Apply one site operator to a node.

        Parameters
        ----------
            loc (int): site number.
            node_op (tn.Node): one site operator.
        """
        node = self.nodes[loc]
        node[str(loc)] ^ node_op[1]
        edges = copy(node.edges)
        edges.remove(node[str(loc)])
        axis_names = copy(node.axis_names)
        axis_names.remove(str(loc))
        applied = tn.contract_between(
            node, node_op, name=node.name, output_edge_order=[node_op[0]] + edges, axis_names=[str(loc)] + axis_names
        )
        self.nodes[loc] = applied

    def replicate_node_dict(self, node_dict, conjugate=False):
        """Replicate dictionary of nodes.

        Parameters
        ----------
            node_dict (dic of tensornetwork.Node): copying dictionary of nodes(e.g. self.nodes)

        Returns
        -------
            dict of tensornetwork.Node: replicated node dictionary
        """
        rep_list = tn.replicate_nodes(node_dict.values(), conjugate=conjugate)
        rep_dict = {node.name: node for node in rep_list}
        return rep_dict

    def expectation_value(self, op, qargs):
        """calculate expectation value of given operator.

        Parameters
        ----------
            op (numpy.ndarray): Expectation value is calculated based on op.
            qargs (list of ints): applied positions of logical qubits.

        Returns
        -------
            expectation_value : float
        """
        state = self.replicate_node_dict(self.nodes)
        dim = int(np.log2(len(op)))
        shape = [2 for _ in range(2 * dim)]
        sites = [self.ptn.output_nodes[i] for i in qargs]
        axis_names_in = ["in" + str(site) for site in sites]
        axis_names_out = [str(site) for site in sites]
        axis_names = axis_names_out + axis_names_in
        node_op = tn.Node(op.reshape(shape), axis_names=axis_names)
        # replicate nodes for calculating expectation value
        rep = self.replicate_node_dict(self.nodes, conjugate=True)
        rep_norm = self.replicate_node_dict(self.nodes)
        rep_norm2 = self.replicate_node_dict(self.nodes, conjugate=True)
        # connect given operators to sites
        concatenated_nodes = set()
        for site in sites:
            concatenated_nodes |= nx.shortest_path(self.graph, site).keys()
        contraction_set = set()
        for site in concatenated_nodes:
            if site in sites:
                node_op["in" + str(site)] ^ state[str(site)][str(site)]
                node_op[str(site)] ^ rep[str(site)][str(site)]
            else:
                state[str(site)][str(site)] ^ rep[str(site)][str(site)]
            contraction_set |= {state[str(site)]} | {rep[str(site)]}
        expectation_value = tn.contractors.auto(list(contraction_set) + [node_op]).tensor
        # calculate norm of TN
        norm_contraction_list = []
        for site in concatenated_nodes:
            rep_norm[str(site)][str(site)] ^ rep_norm2[str(site)][str(site)]
            norm_contraction_list += [rep_norm[str(site)], rep_norm2[str(site)]]
        norm = tn.contractors.auto(norm_contraction_list).tensor
        expectation_value = expectation_value / norm
        return expectation_value

    def expectation_value_ops(self, ops, qargs):
        """calculate expectation value of given operators.
        This command is mainly used for retrieving a probability distribution.

        Parameters
        ----------
            ops (list of numpy.ndarray): Expectation value is calculated based on ops.
                For constructing statevector, ops are projection operators fro each site.
            qargs (list of ints): applied positions of logical qubits.

        Returns
        -------
            expectation value : float
        """
        state = self.replicate_node_dict(self.nodes)
        sites = [self.ptn.output_nodes[i] for i in qargs]
        node_ops = dict()
        for i in range(len(sites)):
            node_op = tn.Node(ops[i], axis_names=["in" + str(sites[i])] + [str(sites[i])])
            node_ops[sites[i]] = node_op
        # replicate nodes for calculating expectation value
        rep = self.replicate_node_dict(self.nodes, conjugate=True)
        rep_norm = self.replicate_node_dict(self.nodes)
        rep_norm2 = self.replicate_node_dict(self.nodes, conjugate=True)
        # connecting given op to sites
        concatenated_nodes = set()
        for site in sites:
            concatenated_nodes |= nx.shortest_path(self.graph, site).keys()
        contraction_set = set()
        for site in concatenated_nodes:
            if site in sites:
                state[str(site)][str(site)] ^ node_ops[site]["in" + str(site)]
                node_ops[site][str(site)] ^ rep[str(site)][str(site)]
            else:
                state[str(site)][str(site)] ^ rep[str(site)][str(site)]
            contraction_set |= {state[str(site)]} | {rep[str(site)]}
        expectation_value = tn.contractors.auto(list(contraction_set) + list(node_ops.values())).tensor
        # calculate norm of TN
        norm_contraction_list = []
        for site in concatenated_nodes:
            rep_norm[str(site)][str(site)] ^ rep_norm2[str(site)][str(site)]
            norm_contraction_list += [rep_norm[str(site)], rep_norm2[str(site)]]
        norm = tn.contractors.auto(norm_contraction_list).tensor
        expectation_value = expectation_value / norm
        return expectation_value

    def get_amplitude(self, number):
        """calculate a probability amplitude of the specified state.

        Parameters
        ----------
            number (int): specifies a state which one wants to know a probability amplitude
            e.g. |0000> corresponds to 0. |1010> corresponds to 10.

        Returns
        -------
            float: the probability amplitude of the specified state.
        """
        proj_to_0 = np.array([[1.0, 0.0], [0.0, 0.0]])
        proj_to_1 = np.array([[0.0, 0.0], [0.0, 1.0]])
        sites = self.ptn.output_nodes
        assert number < 2 ** len(sites)
        ops = []
        for i in range(len(sites)):
            exp = len(sites) - 1 - i
            if (number // 2**exp) == 1:
                op = proj_to_1
                number -= 2**exp
            else:
                op = proj_to_0
            ops.append(op)
        qargs = range(len(sites))
        probability = self.expectation_value_ops(ops, qargs)
        return abs(probability)
