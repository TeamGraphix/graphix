import numpy as np
import quimb.tensor as qtn
from quimb.tensor import Tensor, TensorNetwork
from graphix.clifford import CLIFFORD
from graphix.sim.statevec import meas_op
import string
from copy import deepcopy

VEC = [
    np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]),  # plus
    np.array([1.0 / np.sqrt(2), -1.0 / np.sqrt(2)]),  # minus
    np.array([1.0, 0.0]),  # zero
    np.array([0.0, 1.0]),  # one
    np.array([1.0 / np.sqrt(2), 1.0j / np.sqrt(2)]),  # yplus
    np.array([1.0 / np.sqrt(2), -1.0j / np.sqrt(2)]),  # yminus
]

OP = [
    np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
        ]
    )  # CZ
]

DECOMPOSED_CZ = [
    np.array(
        [
            [[-0.34461337, 1.13818065], [0.0, 0.0]],
            [[0.0, 0.0], [-1.13818065, -0.34461337]],
        ]
    ),  # CZleft
    np.array([[[-1.04849371, 0.0], [0.0, 0.5611368]], [[0.5611368, 0.0], [0.0, 1.04849371]]]),  # CZright
]


class TensorNetworkBackend(TensorNetwork):
    """Tensor Network Simulator for MBQC

    Executes the measurement pattern.
    This class depends on quimb.tensor.core.TensorNetwork.
    """

    def __init__(self, pattern, graph_prep="opt", **kwargs):
        """

        Parameters
        ----------
        pattern : graphix.Pattern
        graph_prep : str
            'sequential' for standard method, 'opt' for faster method
        """
        if "ts" not in kwargs.keys():
            kwargs["ts"] = []
        if isinstance(kwargs["ts"], TensorNetworkBackend):
            super().__init__(**kwargs)
            tnb = kwargs["ts"]
            self.pattern = tnb.pattern
            self.output_nodes = tnb.output_nodes
            self.results = tnb.results
            self.state = tnb.state
            self._dangling = tnb._dangling
            self.__graph_prep = tnb.__graph_prep
            return

        super().__init__(**kwargs)
        self.pattern = pattern
        self.output_nodes = pattern.output_nodes
        self.results = deepcopy(pattern.results)
        self.state = None
        self._dangling = dict()
        self.__graph_prep = graph_prep

    @property
    def graph_prep(self):
        return self.__graph_prep

    @graph_prep.setter
    def graph_prep(self, option):
        if option not in ["opt", "sequential"]:
            raise TypeError(f'{option} is not available. Please choose "sequential" or "opt"')
        self.__graph_prep = option

    def copy(self, deep=False, virtual=False):
        """Copy this object.

        Parameters
        ----------
        deep : bool, optional
            Defaults to False. Whether to copy the underlying data as well.
        virtual : bool, optional
            Defaults to False. To conveniently mimic the behaviour of taking a virtual copy of tensor network, this simply returns `self`. See quimb's document for the detail.

        Returns
        -------
        TensorNetworkBackend :
            duplicated object
        """
        if deep:
            return deepcopy(self)
        return self.__class__(self.pattern, ts=self)

    def add_node(self, node):
        """Add a single node into the network.

        Parameters
        ----------
        node_ind : int
            index of the new node.
        """
        ind = gen_str()
        tag = str(node)
        plus_t = qtn.Tensor(VEC[0], [ind], [tag, "Open"])
        super().add_tensor(plus_t)
        self._dangling[tag] = ind

    def add_nodes(self, nodes):
        """Add nodes into the network

        Parameters
        ----------
        nodes : iterator of int
            index set of the new nodes.
        """
        if self.graph_prep == "sequential":
            for node in nodes:
                ind = gen_str()
                tag = str(node)
                plus_ts = qtn.Tensor(VEC[0], [ind], [tag, "Open"])
                self._dangling[tag] = ind
                super().add_tensor(plus_ts)
        elif self.graph_prep == "opt":
            pass

    def entangle_nodes(self, edge):
        """Make entanglement between nodes specified by edge.

        Parameters
        ----------
        edge : taple of int
            edge specifies two target nodes of the CZ gate.
        """
        if self.graph_prep == "sequential":
            old_inds = [self._dangling[str(node)] for node in edge]
            tids = super()._get_tids_from_inds(old_inds, which="any")
            tensors = [self.tensor_map[tid] for tid in tids]
            new_inds = [gen_str() for _ in range(3)]

            # retag dummy index
            for i in range(2):
                tensors[i].retag({"Open": "Close"}, inplace=True)
                self._dangling[str(edge[i])] = new_inds[i]
            CZ_tn = TensorNetwork(
                [
                    qtn.Tensor(
                        DECOMPOSED_CZ[0],
                        [new_inds[0], old_inds[0], new_inds[2]],
                        [str(edge[0]), "CZ", "Open"],
                    ),
                    qtn.Tensor(
                        DECOMPOSED_CZ[1],
                        [new_inds[2], new_inds[1], old_inds[1]],
                        [str(edge[1]), "CZ", "Open"],
                    ),
                ]
            )

            super().add_tensor_network(CZ_tn)
        elif self.graph_prep == "opt":
            pass

    def initialize(self):
        """Initialize the TN according to the graph preparation strategy."""
        if self.graph_prep == "sequential":
            pass
        elif self.graph_prep == "opt":
            nodes, edges = self.pattern.get_graph()
            self.make_graph_state(nodes, edges)

    def make_graph_state(self, nodes, edges):
        """Prepare the graph state in the efficient way, without directly applying CZ gates.
        This is an internal method of run_cmd().

        Parameters
        ----------
        nodes : iterator of int
            set of the nodes
        edges : iterator of tuple
            set of the edges

        .. seealso:: :meth:`~graphix.sim.tensornet.make_initial()`
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
                super().add_tensor(Tensor(VEC[0], [ind], [str(node), "Open"]))
                continue
            dim_tensor = len(vec_dict[node])
            tensor = np.array(
                [
                    outer_product([VEC[0 + 2 * vec_dict[node][i]] for i in range(dim_tensor)]),
                    outer_product([VEC[1 + 2 * vec_dict[node][i]] for i in range(dim_tensor)]),
                ]
            )
            super().add_tensor(Tensor(tensor, ind_dict[node], [str(node), "Open"]))

    def measure(self, cmd):
        """Perform measurement of the node. In the context of tensornetwork, performing measurement equals to applying measurement operator to the tensor. Here, directly contracted with the projected state.

        Parameters
        ----------
        cmd : list
            measurement command
            i.e. ['M', node, plane angle, s_domain, t_domain]
        """
        # choose the measurement result randomly
        result = np.random.choice([0, 1])
        self.results[cmd[1]] = result

        # extract signals for adaptive angle
        s_signal = np.sum([self.results[j] for j in cmd[4]])
        t_signal = np.sum([self.results[j] for j in cmd[5]])
        angle = cmd[3] * np.pi * (-1) ** s_signal + np.pi * t_signal
        if len(cmd) == 7:
            proj_vec = proj_basis(angle, vop=cmd[6], plane=cmd[2], choice=result)
        else:
            proj_vec = proj_basis(angle, vop=0, plane=cmd[2], choice=result)

        # following procedure tends to keep the norm of the TN
        buffer = 2**0.5
        proj_vec = proj_vec * buffer

        old_ind = self._dangling[str(cmd[1])]
        proj_ts = Tensor(proj_vec, [old_ind], [str(cmd[1]), "M", "Close", "ancilla"]).H

        tid = list(super()._get_tids_from_inds(old_ind))
        tensor = self.tensor_map[tid[0]]
        tensor.retag({"Open": "Close"}, inplace=True)

        super().add_tensor(proj_ts)

    def correct_byproduct(self, cmd):
        """Perform byproduct correction.

        Parameters
        ----------
        cmd : list
            Byproduct command
            i.e.　['X' or 'Z', node, signal_domain]
        """
        if np.mod(np.sum([self.results[j] for j in cmd[2]]), 2) == 1:
            if cmd[0] == "X":
                node_op = np.array([[0.0, 1.0], [1.0, 0.0]])
                gate_type = "X"
            elif cmd[0] == "Z":
                node_op = np.array([[1.0, 0.0], [0.0, -1.0]])
                gate_type = "Z"
            self._apply_one_site_operator(cmd[1], gate_type, node_op)

    def apply_clifford(self, cmd):
        """Apply single-qubit Clifford gate

        Parameters
        ----------
        cmd : list
            clifford command.
            See https://arxiv.org/pdf/2212.11975.pdf for the detail.
        """
        node_op = CLIFFORD[cmd[2]]
        self._apply_one_site_operator(cmd[1], "C", node_op)

    def _apply_one_site_operator(self, node, gate_type, node_op):
        """Internal method for 'measure', 'correct_byproduct', and 'apply_clifford'. Apply one site operator to the node.

        Parameters
        ----------
        node : int
            node index.
        gate_type : str
            gate type of the given operator.
        node_op : numpy.ndarray
            one site operator.
        """
        old_ind = self._dangling[str(node)]
        tid = list(super()._get_tids_from_inds(old_ind))
        tensor = self.tensor_map[tid[0]]

        new_ind = gen_str()
        tensor.retag({"Open": "Close"}, inplace=True)

        node_ts = qtn.Tensor(
            node_op,
            [new_ind, old_ind],
            [str(node), gate_type, "Open"],
        )
        self._dangling[str(node)] = new_ind
        super().add_tensor(node_ts)

    def finalize(self):
        self.state = self

    def simplify(self, *options):
        return super().full_simplify(*options)

    def coef_state(self, number):
        """Calculate the coefficient of the given state.

        Parameters
        ----------
        number : int
            state. e.g. |0000> corresponds to 0. |1010> corresponds to 10.

        Returns
        -------
        complex :
            coefficient
        """
        tn = self.copy()
        # prepare projected state
        for i in range(len(tn.output_nodes)):
            node = str(tn.output_nodes[i])
            exp = len(tn.output_nodes) - i - 1
            if (number // 2**exp) == 1:
                state_out = VEC[3]  # project into |1>
                number -= 2**exp
            else:
                state_out = VEC[2]  # project into |0>
            tensor = Tensor(state_out, [tn._dangling[node]], [node, f"qubit {i}", "Close"])

            # retag
            old_ind = tn._dangling[node]
            tid = list(tn._get_tids_from_inds(old_ind))[0]
            tn.tensor_map[tid].retag({"Open": "Close"})

            tn.add_tensor(tensor)

        # contraction
        tn.simplify()
        coef = tn.contract()

        norm = self.calc_norm()
        return coef / norm**0.5

    def get_amplitude(self, number):
        """Calculate the probability amplitude of the specified state.

        Parameters
        ----------
        number : int
            specifies a state which one wants to know a probability amplitude
            e.g. |0000> corresponds to 0. |1010> corresponds to 10.

        Returns
        -------
        float :
            the probability amplitude of the specified state.
        """
        coef = self.coef_state(number)
        return abs(coef) ** 2

    def to_statevector(self):
        """Retrieve the statevector from the tensornetwork.
        This method requires heavy processing.

        Returns
        -------
        numpy.ndarray :
            statevector
        """
        n_qubit = len(self.output_nodes)
        statevec = np.zeros(2**n_qubit, np.complex128)
        for i in range(len(statevec)):
            statevec[i] = self.coef_state(i)
        return statevec

    def calc_norm(self):
        """Calculate the norm of the state.

        Returns
        -------
        float :
            norm of the state
        """
        tn_cp1 = self.copy()
        tn_cp2 = tn_cp1.conj()
        tn = TensorNetwork([tn_cp1, tn_cp2])
        norm = abs(tn.contract())
        return norm

    def expectation_value(self, op, qargs):
        """Calculate expectation value of the given operator.

        Parameters
        ----------
        op : numpy.ndarray
            Operator to be sandwiched
        qargs : list of int
            Applied positions of logical qubits.

        Returns
        -------
        float :
            Expectation value
        """
        op_dim = len(qargs)
        op = op.reshape([2 for _ in range(2 * op_dim)])
        target_nodes = [self.output_nodes[qarg] for qarg in qargs]
        new_ind_left = [gen_str() for _ in range(op_dim)]
        new_ind_right = [gen_str() for _ in range(op_dim)]

        tn_cp_left = self.copy()
        op_ts = Tensor(op, new_ind_right + new_ind_left, ["Expectation Op.", "Close"])

        tn_cp_right = tn_cp_left.conj()

        # reindex & retag
        for node in self.output_nodes:
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
        tn_cp_left.simplify()
        exp_val = tn_cp_left.contract()
        norm = self.calc_norm()

        return exp_val / norm


def gen_str():
    """Generate dummy string for einsum."""
    result = qtn.rand_uuid()
    return result


def proj_basis(angle, vop, plane, choice):
    """Calculate the projected statevector.

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
        vec = VEC[0 + choice]
        rotU = np.array([[1.0, 0.0], [0, np.e ** (angle * 1.0j)]])
    elif plane == "YZ":
        vec = VEC[4 + choice]
        rotU = np.array(
            [
                [np.cos(angle / 2), -1.0j * np.sin(angle / 2)],
                [-1.0j * np.sin(angle / 2), np.cos(angle / 2)],
            ]
        )
    elif plane == "XZ":
        vec = VEC[2 + choice]
        rotU = np.array(
            [
                [np.cos(angle / 2), -np.sin(angle / 2)],
                [np.sin(angle / 2), np.cos(angle / 2)],
            ]
        )
    vec = np.matmul(rotU, vec)
    vec = np.matmul(CLIFFORD[vop], vec)
    return vec


def outer_product(vectors):
    """Calculate outer product of the given vectors

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
