import numpy as np
import quimb.tensor as qtn
from quimb.tensor import Tensor, TensorNetwork
from graphix.clifford import CLIFFORD, CLIFFORD_MUL, CLIFFORD_CONJ
from graphix.ops import States, Ops
import string
from copy import deepcopy


class TensorNetworkBackend:
    """Tensor Network Simulator for MBQC

    Executes the measurement pattern using TN expression of graph states.
    """

    def __init__(self, pattern, graph_prep="auto", **kwargs):
        """

        Parameters
        ----------
        pattern : graphix.Pattern
        graph_prep : str
            'parallel' :
                Faster method for preparing a graph state.
                The expression of a graph state can be obtained from the graph geometry.
                See https://journals.aps.org/pra/abstract/10.1103/PhysRevA.76.052315 for detail calculation.
                Note that 'N' and 'E' commands in the measurement pattern are ignored.
            'sequential' :
                Sequentially execute N and E commands, strictly following the measurement pattern.
                In this strategy, All N and E commands executed sequentially.
            'auto'(default) :
                Automatically select a preparation strategy based on the max degree of a graph
        **kwargs : Additional keyword args to be passed to quimb.tensor.TensorNetwork.
        """
        self.pattern = pattern
        self.output_nodes = pattern.output_nodes
        self.results = deepcopy(pattern.results)
        if graph_prep in ["parallel", "sequential"]:
            self.graph_prep = graph_prep
        elif graph_prep == "opt":
            self.graph_prep = "parallel"
            print(f"graph preparation strategy '{graph_prep}' is deprecated and will be replaced by 'parallel'")
        elif graph_prep == "auto":
            max_degree = pattern.get_max_degree()
            if max_degree > 5:
                self.graph_prep = "sequential"
            else:
                self.graph_prep = "parallel"
        else:
            raise ValueError(f"Invalid graph preparation strategy: {graph_prep}")

        if self.graph_prep == "parallel":
            if not pattern.is_standard():
                raise ValueError("parallel preparation strategy does not support not-standardized pattern")
            nodes, edges = pattern.get_graph()
            self.state = MBQCTensorNet(
                graph_nodes=nodes,
                graph_edges=edges,
                default_output_nodes=pattern.output_nodes,
                **kwargs,
            )
        elif self.graph_prep == "sequential":
            self.state = MBQCTensorNet(default_output_nodes=pattern.output_nodes, **kwargs)
            self._decomposed_cz = _get_decomposed_cz()
        self._isolated_nodes = pattern.get_isolated_nodes()

    def add_nodes(self, nodes):
        """Add nodes to the network

        Parameters
        ----------
        nodes : iterator of int
            index set of the new nodes.
        """
        if self.graph_prep == "sequential":
            self.state.add_qubits(nodes)
        elif self.graph_prep == "opt":
            pass

    def entangle_nodes(self, edge):
        """Make entanglement between nodes specified by edge.

        Parameters
        ----------
        edge : tuple of int
            edge specifies two target nodes of the CZ gate.
        """
        if self.graph_prep == "sequential":
            old_inds = [self.state._dangling[str(node)] for node in edge]
            tids = self.state._get_tids_from_inds(old_inds, which="any")
            tensors = [self.state.tensor_map[tid] for tid in tids]
            new_inds = [gen_str() for _ in range(3)]

            # retag dummy indices
            for i in range(2):
                tensors[i].retag({"Open": "Close"}, inplace=True)
                self.state._dangling[str(edge[i])] = new_inds[i]
            CZ_tn = TensorNetwork(
                [
                    qtn.Tensor(
                        self._decomposed_cz[0],
                        [new_inds[0], old_inds[0], new_inds[2]],
                        [str(edge[0]), "CZ", "Open"],
                    ),
                    qtn.Tensor(
                        self._decomposed_cz[1],
                        [new_inds[2], new_inds[1], old_inds[1]],
                        [str(edge[1]), "CZ", "Open"],
                    ),
                ]
            )
            self.state.add_tensor_network(CZ_tn)
        elif self.graph_prep == "opt":
            pass

    def measure(self, cmd):
        """Perform measurement of the node. In the context of tensornetwork, performing measurement equals to
        applying measurement operator to the tensor. Here, directly contracted with the projected state.

        Parameters
        ----------
        cmd : list
            measurement command
            i.e. ['M', node, plane angle, s_domain, t_domain]
        """
        if cmd[1] in self._isolated_nodes:
            vector = self.state.get_open_tensor_from_index(cmd[1])
            probs = np.abs(vector) ** 2
            probs = probs / (np.sum(probs))
            result = np.random.choice([0, 1], p=probs)
            self.results[cmd[1]] = result
            buffer = 1 / probs[result] ** 0.5
        else:
            # choose the measurement result randomly
            result = np.random.choice([0, 1])
            self.results[cmd[1]] = result
            buffer = 2**0.5

        # extract signals for adaptive angle
        s_signal = np.sum([self.results[j] for j in cmd[4]])
        t_signal = np.sum([self.results[j] for j in cmd[5]])
        angle = cmd[3] * np.pi
        if len(cmd) == 7:
            vop = cmd[6]
        else:
            vop = 0
        if int(s_signal % 2) == 1:
            vop = CLIFFORD_MUL[1, vop]
        if int(t_signal % 2) == 1:
            vop = CLIFFORD_MUL[3, vop]
        proj_vec = proj_basis(angle, vop=vop, plane=cmd[2], choice=result)

        # buffer is necessary for maintaing the norm invariant
        proj_vec = proj_vec * buffer
        self.state.measure_single(cmd[1], basis=proj_vec)

    def correct_byproduct(self, cmd):
        """Perform byproduct correction.

        Parameters
        ----------
        cmd : list
            Byproduct command
            i.e. ['X' or 'Z', node, signal_domain]
        """
        if np.mod(np.sum([self.results[j] for j in cmd[2]]), 2) == 1:
            if cmd[0] == "X":
                self.state.evolve_single(cmd[1], Ops.x, "X")
            elif cmd[0] == "Z":
                self.state.evolve_single(cmd[1], Ops.z, "Z")

    def apply_clifford(self, cmd):
        """Apply single-qubit Clifford gate

        Parameters
        ----------
        cmd : list
            clifford command.
            See https://arxiv.org/pdf/2212.11975.pdf for the detail.
        """
        node_op = CLIFFORD[cmd[2]]
        self.state.evolve_single(cmd[1], node_op, "C")

    def finalize(self):
        pass


class MBQCTensorNet(TensorNetwork):
    """Tensor Network Simulator interface for MBQC patterns, using quimb.tensor.core.TensorNetwork."""

    def __init__(
        self,
        graph_nodes=None,
        graph_edges=None,
        default_output_nodes=None,
        ts=[],
        **kwargs,
    ):
        """
        Initialize MBQCTensorNet.

        Parameters
        ----------
        graph_nodes (optional): list of int
            node indices of the graph state.
        graph_edges (optional) : list of tuple of int
            edge indices of the graph state.
        default_output_nodes : list of int
            output node indices at the end of MBQC operations, if known in advance.
        ts (optional): quimb.tensor.core.TensorNetwork or empty list
            optional initial state.
        """
        if isinstance(ts, MBQCTensorNet):
            super().__init__(ts=ts, **kwargs)
            self._dangling = ts._dangling
            self.default_output_nodes = default_output_nodes
        else:
            super().__init__(ts=ts, **kwargs)
            self._dangling = dict()
            self.default_output_nodes = default_output_nodes
        # prepare the graph state if graph_nodes and graph_edges are given
        if graph_nodes is not None and graph_edges is not None:
            self.set_graph_state(graph_nodes, graph_edges)

    def get_open_tensor_from_index(self, index):
        """Get tensor specified by node index. The tensor has a dangling edge.

        Parameters
        ----------
        index : str
            node index

        Returns
        -------
        numpy.ndarray :
            Specified tensor
        """
        if isinstance(index, int):
            index = str(index)
        assert isinstance(index, str)
        tags = [index, "Open"]
        tid = list(self._get_tids_from_tags(tags, which="all"))[0]
        tensor = self.tensor_map[tid]
        return tensor.data

    def add_qubit(self, index, state="plus"):
        """Add a single qubit to the network.

        Parameters
        ----------
        index : int
            index of the new qubit.
        state (optional): str or 2-element np.ndarray
            initial state of the new qubit.
            "plus", "minus", "zero", "one", "iplus", "iminus", or 1*2 np.ndarray (arbitrary state).
        """
        ind = gen_str()
        tag = str(index)
        if state == "plus":
            vec = States.plus
        elif state == "minus":
            vec = States.minus
        elif state == "zero":
            vec = States.zero
        elif state == "one":
            vec = States.one
        elif state == "iplus":
            vec = States.iplus
        elif state == "iminus":
            vec = States.iminus
        else:
            assert state.shape == (2,), "state must be 2-element np.ndarray"
            assert np.isclose(np.linalg.norm(state), 1), "state must be normalized"
            vec = state
        tsr = qtn.Tensor(vec, [ind], [tag, "Open"])
        self.add_tensor(tsr)
        self._dangling[tag] = ind

    def evolve_single(self, index, arr, label="U"):
        """Apply single-qubit operator to a qubit with the given index.

        Parameters
        ----------
        index : int
            qubit index.
        arr : 2*2 numpy.ndarray
            single-qubit operator.
        label (optional): str
            label for the gate.
        """
        old_ind = self._dangling[str(index)]
        tid = list(self._get_tids_from_inds(old_ind))
        tensor = self.tensor_map[tid[0]]

        new_ind = gen_str()
        tensor.retag({"Open": "Close"}, inplace=True)

        node_ts = qtn.Tensor(
            arr,
            [new_ind, old_ind],
            [str(index), label, "Open"],
        )
        self._dangling[str(index)] = new_ind
        self.add_tensor(node_ts)

    def add_qubits(self, indices, states="plus"):
        """Add qubits to the network

        Parameters
        ----------
        indices : iterator of int
            indices of the new qubits.
        states (optional): str or 2*2 numpy.ndarray or list
            initial states of the new qubits.
            "plus", "minus", "zero", "one", "iplus", "iminus", or 1*2 np.ndarray (arbitrary state).
            list of the above, to specify the initial state of each qubit.
        """
        if type(states) != list:
            states = [states] * len(indices)
        for i, ind in enumerate(indices):
            self.add_qubit(ind, state=states[i])

    def measure_single(self, index, basis="Z", bypass_probability_calculation=True, outcome=None):
        """Measure a node in specified basis. Note this does not perform the partial trace.

        Parameters
        ----------
        index : int
            index of the node to be measured.
        basis : str or np.ndarray
            default "Z".
            measurement basis, "Z" or "X" or "Y" for Pauli basis measurements.
            1*2 numpy.ndarray for arbitrary measurement bases.
        bypass_probability_calculation : bool
            default True.
            if True, skip the calculation of the probability of the measurement
            result and use equal probability for each result.
            if False, calculate the probability of the measurement result from the state.
        outcome : int (0 or 1)
            User-chosen measurement result, giving the outcome of (-1)^{outcome}.

        Returns
        -------
        int
            measurement result.
        """
        if bypass_probability_calculation:
            if outcome is not None:
                result = outcome
            else:
                result = np.random.choice([0, 1])
            # Basis state to be projected
            if type(basis) == np.ndarray:
                if outcome is not None:
                    raise Warning("Measurement outcome is chosen but the basis state was given.")
                proj_vec = basis
            elif basis == "Z" and result == 0:
                proj_vec = States.zero
            elif basis == "Z" and result == 1:
                proj_vec = States.one
            elif basis == "X" and result == 0:
                proj_vec = States.plus
            elif basis == "X" and result == 1:
                proj_vec = States.minus
            elif basis == "Y" and result == 0:
                proj_vec = States.iplus
            elif basis == "Y" and result == 1:
                proj_vec = States.iminus
            else:
                raise ValueError("Invalid measurement basis.")
        else:
            raise NotImplementedError("Measurement probability calculation not implemented.")
        old_ind = self._dangling[str(index)]
        proj_ts = Tensor(proj_vec, [old_ind], [str(index), "M", "Close", "ancilla"]).H
        # add the tensor to the network
        tid = list(self._get_tids_from_inds(old_ind))
        tensor = self.tensor_map[tid[0]]
        tensor.retag({"Open": "Close"}, inplace=True)
        self.add_tensor(proj_ts)
        return result

    def set_graph_state(self, nodes, edges):
        """Prepare the graph state without directly applying CZ gates.

        Parameters
        ----------
        nodes : iterator of int
            set of the nodes
        edges : iterator of tuple
            set of the edges

        .. seealso:: :meth:`~graphix.sim.tensornet.TensorNetworkBackend.__init__()`
        """
        ind_dict = dict()
        vec_dict = dict()
        for edge in edges:
            for node in edge:
                if node not in ind_dict.keys():
                    ind = gen_str()
                    self._dangling[str(node)] = ind
                    ind_dict[node] = [ind]
                    vec_dict[node] = []
            greater = edge[0] > edge[1]  # true for 1/0, false for +/-
            vec_dict[edge[0]].append(greater)
            vec_dict[edge[1]].append(not greater)

            ind = gen_str()
            ind_dict[edge[0]].append(ind)
            ind_dict[edge[1]].append(ind)

        for node in nodes:
            if node not in ind_dict.keys():
                ind = gen_str()
                self._dangling[str(node)] = ind
                self.add_tensor(Tensor(States.plus, [ind], [str(node), "Open"]))
                continue
            dim_tensor = len(vec_dict[node])
            tensor = np.array(
                [
                    outer_product([States.vec[0 + 2 * vec_dict[node][i]] for i in range(dim_tensor)]),
                    outer_product([States.vec[1 + 2 * vec_dict[node][i]] for i in range(dim_tensor)]),
                ]
            ) * 2 ** (dim_tensor / 4 - 1.0 / 2)
            self.add_tensor(Tensor(tensor, ind_dict[node], [str(node), "Open"]))

    def get_basis_coefficient(self, basis, normalize=True, indices=None, **kwagrs):
        """Calculate the coefficient of a given computational basis.

        Parameters
        ----------
        basis : int or str
            computational basis expressed in binary (str) or integer, e.g. 101 or 5.
        normalize (optional): bool
            if True, normalize the coefficient by the norm of the entire state.
        indices (optional): list of int
            target qubit indices to compute the coefficients, default is the MBQC output nodes (self.default_output_nodes).

        Returns
        -------
        coef : complex
            coefficient
        """
        if indices == None:
            indices = self.default_output_nodes
        if isinstance(basis, str):
            basis = int(basis, 2)
        tn = self.copy()
        # prepare projected state
        for i in range(len(indices)):
            node = str(indices[i])
            exp = len(indices) - i - 1
            if (basis // 2**exp) == 1:
                state_out = States.one  # project onto |1>
                basis -= 2**exp
            else:
                state_out = States.zero  # project onto |0>
            tensor = Tensor(state_out, [tn._dangling[node]], [node, f"qubit {i}", "Close"])
            # retag
            old_ind = tn._dangling[node]
            tid = list(tn._get_tids_from_inds(old_ind))[0]
            tn.tensor_map[tid].retag({"Open": "Close"})
            tn.add_tensor(tensor)

        # contraction
        tn_simplified = tn.full_simplify("ADCR")
        coef = tn_simplified.contract(output_inds=[], **kwagrs)
        if normalize:
            norm = self.get_norm()
            return coef / norm
        else:
            return coef

    def get_basis_amplitude(self, basis, **kwagrs):
        """Calculate the probability amplitude of the specified computational basis state.

        Parameters
        ----------
        basis : int or str
            computational basis expressed in binary (str) or integer, e.g. 101 or 5.

        Returns
        -------
        float :
            the probability amplitude of the specified state.
        """
        if isinstance(basis, str):
            basis = int(basis, 2)
        coef = self.get_basis_coefficient(basis, **kwagrs)
        return abs(coef) ** 2

    def to_statevector(self, indices=None, **kwagrs):
        """Retrieve the statevector from the tensornetwork.
        This method tends to be slow however we plan to parallelize this.

        Parameters
        ----------
        indices (optional): list of int
            target qubit indices. Default is the MBQC output nodes (self.default_output_nodes).

        Returns
        -------
        numpy.ndarray :
            statevector
        """
        if indices == None:
            n_qubit = len(self.default_output_nodes)
        else:
            n_qubit = len(indices)
        statevec = np.zeros(2**n_qubit, np.complex128)
        for i in range(len(statevec)):
            statevec[i] = self.get_basis_coefficient(i, normalize=False, indices=indices, **kwagrs)
        return statevec / np.linalg.norm(statevec)

    def get_norm(self, **kwagrs):
        """Calculate the norm of the state.

        Returns
        -------
        float :
            norm of the state
        """
        tn_cp1 = self.copy()
        tn_cp2 = tn_cp1.conj()
        tn = TensorNetwork([tn_cp1, tn_cp2])
        tn_simplified = tn.full_simplify("ADCR")
        norm = abs(tn_simplified.contract(output_inds=[], **kwagrs)) ** 0.5
        return norm

    def expectation_value(self, op, qubit_indices, output_node_indices=None, **kwagrs):
        """Calculate expectation value of the given operator,

        Parameters
        ----------
        op : numpy.ndarray
            single- or multi-qubit Hermitian operator
        qubit_indices : list of int
            Applied positions of **logical** qubits.
        output_node_indices (optional): list of int
            Indices of nodes in the entire TN, that remain unmeasured after MBQC operations.
            Default is the output nodes specified in measurement pattern (self.default_output_nodes).

        Returns
        -------
        float :
            Expectation value
        """
        if output_node_indices == None:
            if self.default_output_nodes == None:
                raise ValueError("output_nodes is not set.")
            else:
                target_nodes = [self.default_output_nodes[ind] for ind in qubit_indices]
                out_inds = self.default_output_nodes
        else:
            target_nodes = [output_node_indices[ind] for ind in qubit_indices]
            out_inds = output_node_indices
        op_dim = len(qubit_indices)
        op = op.reshape([2 for _ in range(2 * op_dim)])
        new_ind_left = [gen_str() for _ in range(op_dim)]
        new_ind_right = [gen_str() for _ in range(op_dim)]
        tn_cp_left = self.copy()
        op_ts = Tensor(op, new_ind_right + new_ind_left, ["Expectation Op.", "Close"])
        tn_cp_right = tn_cp_left.conj()

        # reindex & retag
        for node in out_inds:
            old_ind = tn_cp_left._dangling[str(node)]
            tid_left = list(tn_cp_left._get_tids_from_inds(old_ind))[0]
            tid_right = list(tn_cp_right._get_tids_from_inds(old_ind))[0]
            if node in target_nodes:
                tn_cp_left.tensor_map[tid_left].reindex({old_ind: new_ind_left[target_nodes.index(node)]}, inplace=True)
                tn_cp_right.tensor_map[tid_right].reindex(
                    {old_ind: new_ind_right[target_nodes.index(node)]}, inplace=True
                )
            tn_cp_left.tensor_map[tid_left].retag({"Open": "Close"})
            tn_cp_right.tensor_map[tid_right].retag({"Open": "Close"})
        tn_cp_left.add([op_ts, tn_cp_right])

        # contraction
        tn_cp_left = tn_cp_left.full_simplify("ADCR")
        exp_val = tn_cp_left.contract(output_inds=[], **kwagrs)
        norm = self.get_norm(**kwagrs)
        return exp_val / norm**2

    def evolve(self, operator, qubit_indices, decompose=True, **kwagrs):
        """Apply an arbitrary operator to the state.

        Parameters
        ----------
        operator : numpy.ndarray
            operator.
        qubit_indices : list of int
            Applied positions of **logical** qubits.
        decompose : bool, optional
            default True
            whether a given operator will be decomposed or not. If True, operator is decomposed into Matrix Product Operator(MPO)
        """
        if len(operator.shape) != len(qubit_indices) * 2:
            shape = [2 for _ in range(2 * len(qubit_indices))]
            operator = operator.reshape(shape)

        # operator indices
        node_indices = [self.default_output_nodes[index] for index in qubit_indices]
        old_ind_list = [self._dangling[str(index)] for index in node_indices]
        new_ind_list = [gen_str() for _ in range(len(node_indices))]
        for i in range(len(node_indices)):
            self._dangling[str(node_indices[i])] = new_ind_list[i]

        ts = Tensor(
            operator,
            new_ind_list + old_ind_list,
            [str(index) for index in node_indices],
        )
        if decompose:  # decompose tensor into Matrix Product Operator(MPO)
            tensors = []
            bond_inds = {0: None}
            for i in range(len(node_indices) - 1):
                bond_inds[i + 1] = gen_str()
                left_inds = [new_ind_list[i], old_ind_list[i]]
                if bond_inds[i]:
                    left_inds.append(bond_inds[i])
                unit_tensor, ts = ts.split(left_inds=left_inds, bond_ind=bond_inds[i + 1], **kwagrs)
                tensors.append(unit_tensor)
            tensors.append(ts)
            op_tensor = TensorNetwork(tensors)
        else:
            op_tensor = ts
        self.add(op_tensor)

    def copy(self, deep=False):
        """Returns the copy of this object.

        Parameters
        ----------
        deep : bool, optional
            Defaults to False.
            Whether to copy the underlying data as well.

        Returns
        -------
        TensorNetworkBackend :
            duplicated object
        """
        if deep:
            return deepcopy(self)
        else:
            return self.__class__(ts=self)


def _get_decomposed_cz():
    """Returns the decomposed cz tensors. This is an internal method.

    CZ gate can be decomposed into two 3-rank tensors(Schmidt rank = 2).
    Decomposing into low-rank tensors is important preprocessing for
    the optimal contraction path searching problem.
    So, in this backend, the DECOMPOSED_CZ gate is applied
    instead of the original CZ gate.

        Decomposing CZ gate

        output            output
        |    |           |      |
       --------   SVD   ---    ---
       |  CZ  |   -->   |L|----|R|
       --------         ---    ---
        |    |           |      |
        input             input

    4-rank x1         3-rank x2
    """
    cz_ts = Tensor(
        Ops.cz.reshape((2, 2, 2, 2)).astype(np.float64),
        ["O1", "O2", "I1", "I2"],
        ["CZ"],
    )
    decomposed_cz = cz_ts.split(left_inds=["O1", "I1"], right_inds=["O2", "I2"], max_bond=4)
    return [
        decomposed_cz.tensors[0].data,
        decomposed_cz.tensors[1].data,
    ]


def gen_str():
    """Generate dummy string for einsum."""
    result = qtn.rand_uuid()
    return result


def proj_basis(angle, vop, plane, choice):
    """the projected statevector.

    Parameters
    ----------
    angle : float
        measurement angle
    vop : int
        CLIFFORD index
    plane : str
        measurement plane
    choice : int
        measurement result

    Returns
    -------
    numpy.ndarray :
        projected state
    """
    if plane == "XY":
        vec = States.vec[0 + choice]
        rotU = Ops.Rz(angle)
    elif plane == "YZ":
        vec = States.vec[4 + choice]
        rotU = Ops.Rx(angle)
    elif plane == "XZ":
        vec = States.vec[0 + choice]
        rotU = Ops.Ry(-angle)
    vec = np.matmul(rotU, vec)
    vec = np.matmul(CLIFFORD[CLIFFORD_CONJ[vop]], vec)
    return vec


def outer_product(vectors):
    """outer product of the given vectors

    Parameters
    ----------
    vectors : list of vector
        vectors

    Returns
    -------
    numpy.ndarray :
        tensor object.
    """
    subscripts = string.ascii_letters[: len(vectors)]
    subscripts = ",".join(subscripts) + "->" + subscripts
    return np.einsum(subscripts, *vectors)
