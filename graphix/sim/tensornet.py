"""Tensor Network Simulator for MBQC."""

from __future__ import annotations

import string
import warnings
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, SupportsComplex, TypeAlias

import numpy as np
import numpy.typing as npt
import quimb.tensor as qtn
from quimb.tensor import Tensor, TensorNetwork

# override introduced in Python 3.12
from typing_extensions import override

from graphix import command
from graphix.branch_selector import BranchSelector, RandomBranchSelector
from graphix.ops import Ops
from graphix.parameter import Expression
from graphix.sim.base_backend import Backend
from graphix.states import BasicStates, PlanarState

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from cotengra.oe import PathOptimizer
    from numpy.random import Generator

    from graphix import Pattern
    from graphix.clifford import Clifford
    from graphix.measurements import Measurement, Outcome
    from graphix.sim import Data
    from graphix.simulator import MeasureMethod

PrepareState: TypeAlias = str | npt.NDArray[np.complex128]


class MBQCTensorNet(TensorNetwork):
    """Tensor Network Simulator interface for MBQC patterns, using quimb.tensor.core.TensorNetwork."""

    _dangling: dict[str, str]

    def __init__(
        self,
        branch_selector: BranchSelector,
        graph_nodes: Iterable[int] | None = None,
        graph_edges: Iterable[tuple[int, int]] | None = None,
        default_output_nodes: Iterable[int] | None = None,
        ts: list[TensorNetwork] | TensorNetwork | None = None,
        virtual: bool = False,
    ) -> None:
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
        if ts is None:
            ts = []
        super().__init__(ts=ts, virtual=virtual)
        self._dangling = ts._dangling if isinstance(ts, MBQCTensorNet) else {}
        self.default_output_nodes = None if default_output_nodes is None else list(default_output_nodes)
        # prepare the graph state if graph_nodes and graph_edges are given
        if graph_nodes is not None and graph_edges is not None:
            self.prepare_graph_state(graph_nodes, graph_edges)
        self.__branch_selector = branch_selector

    def open_tensor(self, index: int | str) -> npt.NDArray[np.complex128]:
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
        tid = next(iter(self._get_tids_from_tags(tags, which="all")))
        tensor = self.tensor_map[tid]
        return tensor.data.astype(dtype=np.complex128)

    def add_qubit(self, index: int, state: PrepareState = "plus") -> None:
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
            vec = BasicStates.PLUS.to_statevector()
        elif state == "minus":
            vec = BasicStates.MINUS.to_statevector()
        elif state == "zero":
            vec = BasicStates.ZERO.to_statevector()
        elif state == "one":
            vec = BasicStates.ONE.to_statevector()
        elif state == "iplus":
            vec = BasicStates.PLUS_I.to_statevector()
        elif state == "iminus":
            vec = BasicStates.MINUS_I.to_statevector()
        else:
            if isinstance(state, str):
                raise TypeError(f"Unknown state: {state}")
            if state.shape != (2,):
                raise ValueError("state must be 2-element np.ndarray")
            if not np.isclose(np.linalg.norm(state), 1):
                raise ValueError("state must be normalized")
            vec = state
        tsr = Tensor(vec, [ind], [tag, "Open"])
        self.add_tensor(tsr)
        self._dangling[tag] = ind

    def evolve_single(self, index: int, arr: npt.NDArray[np.complex128], label: str = "U") -> None:
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

        node_ts = Tensor(
            arr,
            [new_ind, old_ind],
            [str(index), label, "Open"],
        )
        self._dangling[str(index)] = new_ind
        self.add_tensor(node_ts)

    def add_qubits(self, indices: Sequence[int], states: PrepareState | Iterable[PrepareState] = "plus") -> None:
        """Add qubits to the network.

        Parameters
        ----------
        indices : iterator of int
            indices of the new qubits.
        states (optional): Data
            initial state or list of initial states of the new qubits.
        """
        if isinstance(states, str):
            states_iter: list[PrepareState] = [states] * len(indices)
        else:
            states_list = list(states)
            # `states` is of type `PrepareState`, a type alias for
            # `str | npt.NDArray[np.complex128]`. To distinguish
            # between the two cases, we just need to check whether
            # an element is a character or a complex number.
            if len(states_list) == 0 or isinstance(states_list[0], SupportsComplex):
                states_iter = [np.array(states_list)] * len(indices)
            else:

                def get_prepare_state(item: PrepareState | SupportsComplex) -> PrepareState:
                    if isinstance(item, SupportsComplex):
                        raise TypeError("Unexpected complex")
                    return item

                states_iter = [get_prepare_state(item) for item in states_list]
        for ind, state in zip(indices, states_iter, strict=True):
            self.add_qubit(ind, state)

    def measure_single(
        self,
        index: int,
        basis: str | npt.NDArray[np.complex128] = "Z",
        bypass_probability_calculation: bool = True,
        outcome: Outcome | None = None,
        rng: Generator | None = None,
    ) -> Outcome:
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
            result = outcome if outcome is not None else self.__branch_selector.measure(index, lambda: 0.5, rng=rng)
            # Basis state to be projected
            if isinstance(basis, np.ndarray):
                if outcome is not None:
                    raise Warning("Measurement outcome is chosen but the basis state was given.")
                proj_vec = basis
            elif basis == "Z" and result == 0:
                proj_vec = BasicStates.ZERO.to_statevector()
            elif basis == "Z" and result == 1:
                proj_vec = BasicStates.ONE.to_statevector()
            elif basis == "X" and result == 0:
                proj_vec = BasicStates.PLUS.to_statevector()
            elif basis == "X" and result == 1:
                proj_vec = BasicStates.MINUS.to_statevector()
            elif basis == "Y" and result == 0:
                proj_vec = BasicStates.PLUS_I.to_statevector()
            elif basis == "Y" and result == 1:
                proj_vec = BasicStates.MINUS_I.to_statevector()
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

    def prepare_graph_state(self, nodes: Iterable[int], edges: Iterable[tuple[int, int]]) -> None:
        """Prepare the graph state without directly applying CZ gates.

        Parameters
        ----------
        nodes : iterator of int
            set of the nodes
        edges : iterator of tuple
            set of the edges

        .. seealso:: :meth:`~graphix.sim.tensornet.TensorNetworkBackend.__init__()`
        """
        ind_dict: dict[int, list[str]] = {}
        vec_dict: dict[int, list[bool]] = {}
        for edge in edges:
            for node in edge:
                if node not in ind_dict:
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
            if node not in ind_dict:
                ind = gen_str()
                self._dangling[str(node)] = ind
                self.add_tensor(Tensor(BasicStates.PLUS.to_statevector(), [ind], [str(node), "Open"]))
                continue
            dim_tensor = len(vec_dict[node])
            tensor = np.array(
                [
                    outer_product(
                        [BasicStates.VEC[0 + 2 * vec_dict[node][i]].to_statevector() for i in range(dim_tensor)]
                    ),
                    outer_product(
                        [BasicStates.VEC[1 + 2 * vec_dict[node][i]].to_statevector() for i in range(dim_tensor)]
                    ),
                ]
            ) * 2 ** (dim_tensor / 4 - 1.0 / 2)
            self.add_tensor(Tensor(tensor, ind_dict[node], [str(node), "Open"]))

    def _require_default_output_nodes(self) -> list[int]:
        if self.default_output_nodes is None:
            raise ValueError("output_nodes is not set.")
        return self.default_output_nodes

    def basis_coefficient(
        self, basis: int | str, normalize: bool = True, indices: Sequence[int] | None = None
    ) -> complex:
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
        if indices is None:
            indices = self._require_default_output_nodes()
        if isinstance(basis, str):
            basis = int(basis, 2)
        tn = self.copy()
        # prepare projected state
        for i in range(len(indices)):
            node = str(indices[i])
            exp = len(indices) - i - 1
            if (basis // 2**exp) == 1:
                state_out = BasicStates.ONE.to_statevector()  # project onto |1>
                basis -= 2**exp
            else:
                state_out = BasicStates.ZERO.to_statevector()  # project onto |0>
            tensor = Tensor(state_out, [tn._dangling[node]], [node, f"qubit {i}", "Close"])
            # retag
            old_ind = tn._dangling[node]
            tid = next(iter(tn._get_tids_from_inds(old_ind)))
            tn.tensor_map[tid].retag({"Open": "Close"})
            tn.add_tensor(tensor)

        # contraction
        tn_simplified = tn.full_simplify("ADCR")
        coef = tn_simplified.contract(output_inds=[])
        if normalize:
            norm = self.norm()
            return coef / norm
        return coef

    def basis_amplitude(self, basis: str | int) -> float:
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
        coef = self.basis_coefficient(basis)
        return abs(coef) ** 2

    def to_statevector(self, indices: Sequence[int] | None = None) -> npt.NDArray[np.complex128]:
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
        n_qubit = len(self._require_default_output_nodes()) if indices is None else len(indices)
        statevec: npt.NDArray[np.complex128] = np.zeros(2**n_qubit, np.complex128)
        for i in range(len(statevec)):
            statevec[i] = self.basis_coefficient(i, normalize=False, indices=indices)
        return statevec / np.linalg.norm(statevec)

    def flatten(self) -> npt.NDArray[np.complex128]:
        """Return flattened statevector."""
        return self.to_statevector().flatten()

    def norm(self, optimize: str | PathOptimizer | None = None) -> float:
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
        contraction = tn_simplified.contract(output_inds=[], optimize=optimize)
        return float(abs(contraction) ** 0.5)

    def expectation_value(
        self,
        op: npt.NDArray[np.complex128],
        qubit_indices: Sequence[int],
        output_node_indices: Iterable[int] | None = None,
        optimize: str | PathOptimizer | None = None,
    ) -> float:
        """Calculate expectation value of the given operator.

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
        out_inds = self._require_default_output_nodes() if output_node_indices is None else list(output_node_indices)
        target_nodes = [out_inds[ind] for ind in qubit_indices]
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
            tid_left = next(iter(tn_cp_left._get_tids_from_inds(old_ind)))
            tid_right = next(iter(tn_cp_right._get_tids_from_inds(old_ind)))
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
        exp_val = tn_cp_left.contract(output_inds=[], optimize=optimize)
        norm = self.norm(optimize=optimize)
        return exp_val / norm**2

    def evolve(self, operator: npt.NDArray[np.complex128], qubit_indices: list[int], decompose: bool = True) -> None:
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
        default_output_nodes = self._require_default_output_nodes()
        node_indices = [default_output_nodes[index] for index in qubit_indices]
        old_ind_list = [self._dangling[str(index)] for index in node_indices]
        new_ind_list = [gen_str() for _ in range(len(node_indices))]
        for i in range(len(node_indices)):
            self._dangling[str(node_indices[i])] = new_ind_list[i]

        ts: Tensor | TensorNetwork = Tensor(
            operator,
            new_ind_list + old_ind_list,
            [str(index) for index in node_indices],
        )
        if decompose:  # decompose tensor into Matrix Product Operator(MPO)
            tensors: list[Tensor | TensorNetwork] = []
            bond_inds: dict[int, str | None] = {0: None}
            for i in range(len(node_indices) - 1):
                bond_inds[i + 1] = gen_str()
                left_inds: list[str] = [new_ind_list[i], old_ind_list[i]]
                bond_ind = bond_inds[i]
                if bond_ind is not None:
                    left_inds.append(bond_ind)
                unit_tensor, ts = ts.split(left_inds=left_inds, bond_ind=bond_inds[i + 1])
                tensors.append(unit_tensor)
            tensors.append(ts)
            ts = TensorNetwork(tensors)
        self.add(ts)

    @override
    def copy(self, virtual: bool = False, deep: bool = False) -> MBQCTensorNet:
        """Return the copy of this object.

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
        return self.__class__(branch_selector=self.__branch_selector, ts=self)


def _decompose_cz() -> list[npt.NDArray[np.complex128]]:
    """Return the decomposed cz tensors.

    This is an internal method.

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
        Ops.CZ.reshape((2, 2, 2, 2)).astype(np.complex128),
        ["O1", "O2", "I1", "I2"],
        ["CZ"],
    )
    decomposed_cz = cz_ts.split(left_inds=["O1", "I1"], right_inds=["O2", "I2"], max_bond=4)
    return [
        decomposed_cz.tensors[0].data.astype(np.complex128),
        decomposed_cz.tensors[1].data.astype(np.complex128),
    ]


@dataclass(frozen=True)
class _AbstractTensorNetworkBackend(Backend[MBQCTensorNet], ABC):
    state: MBQCTensorNet
    pattern: Pattern
    graph_prep: str
    input_state: Data
    branch_selector: BranchSelector
    output_nodes: list[int]
    results: dict[int, Outcome]
    _decomposed_cz: list[npt.NDArray[np.complex128]]
    _isolated_nodes: set[int]


@dataclass(frozen=True)
class TensorNetworkBackend(_AbstractTensorNetworkBackend):
    """Tensor Network Simulator for MBQC.

    Executes the measurement pattern using TN expression of graph states.

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
    input_state : preparation for input states (only BasicStates.PLUS is supported for tensor networks yet),
    branch_selector: :class:`graphix.branch_selector.BranchSelector`, optional
        Branch selector to be used for measurements.
    """

    def __init__(
        self,
        pattern: Pattern,
        graph_prep: str = "auto",
        input_state: Data | None = None,
        branch_selector: BranchSelector | None = None,
    ) -> None:
        """Construct a tensor network backend."""
        if input_state is None:
            input_state = BasicStates.PLUS
        elif input_state != BasicStates.PLUS:
            msg = "TensorNetworkBackend currently only supports BasicStates.PLUS as input state."
            raise NotImplementedError(msg)
        if branch_selector is None:
            branch_selector = RandomBranchSelector()
        if graph_prep in {"parallel", "sequential"}:
            pass
        elif graph_prep == "opt":
            graph_prep = "parallel"
            warnings.warn(
                f"graph preparation strategy '{graph_prep}' is deprecated and will be replaced by 'parallel'",
                stacklevel=1,
            )
        elif graph_prep == "auto":
            max_degree = pattern.compute_max_degree()
            # "parallel" does not support non standard pattern
            graph_prep = "sequential" if max_degree > 5 or not pattern.is_standard() else "parallel"
        else:
            raise ValueError(f"Invalid graph preparation strategy: {graph_prep}")
        results = deepcopy(pattern.results)
        if graph_prep == "parallel":
            if not pattern.is_standard():
                raise ValueError("parallel preparation strategy does not support not-standardized pattern")
            graph = pattern.extract_graph()
            state = MBQCTensorNet(
                graph_nodes=graph.nodes,
                graph_edges=graph.edges,
                default_output_nodes=pattern.output_nodes,
                branch_selector=branch_selector,
            )
            decomposed_cz = []
        else:  # graph_prep == "sequential":
            state = MBQCTensorNet(default_output_nodes=pattern.output_nodes, branch_selector=branch_selector)
            decomposed_cz = _decompose_cz()
        isolated_nodes = pattern.extract_isolated_nodes()
        super().__init__(
            state,
            pattern,
            graph_prep,
            input_state,
            branch_selector,
            pattern.output_nodes,
            results,
            decomposed_cz,
            isolated_nodes,
        )

    @override
    def add_nodes(self, nodes: Sequence[int], data: Data = BasicStates.PLUS) -> None:
        """
        Add new nodes (qubits) to the network and initialize them in a specified state.

        Parameters
        ----------
        nodes : Sequence[int]
            A list of node indices to add to the backend. These indices can be any
            integer values but must be fresh: each index must be distinct from all
            previously added nodes.

        data : Data, optional
            The state in which to initialize the newly added nodes.

            - If a single basic state is provided, all new nodes are initialized in that state.
            - If a list of basic states is provided, it must match the length of ``nodes``, and
              each node is initialized with its corresponding state.

        Notes
        -----
        Previously existing nodes remain unchanged.
        """
        if data != BasicStates.PLUS:
            raise NotImplementedError(
                "TensorNetworkBackend currently only supports |+> input state (see https://github.com/TeamGraphix/graphix/issues/167)."
            )
        if self.graph_prep == "sequential":
            self.state.add_qubits(nodes)
        elif self.graph_prep == "opt":
            pass

    @override
    def entangle_nodes(self, edge: tuple[int, int]) -> None:
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
            cz_tn = TensorNetwork(
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
            self.state.add_tensor_network(cz_tn)
        elif self.graph_prep == "opt":
            pass

    @override
    def measure(self, node: int, measurement: Measurement, rng: Generator | None = None) -> Outcome:
        """Perform measurement of the node.

        In the context of tensornetwork, performing measurement equals to
        applying measurement operator to the tensor. Here, directly contracted with the projected state.

        Parameters
        ----------
        node : int
            index of the node to measure
        measurement : Measurement
            measure plane and angle
        """
        if node in self._isolated_nodes:
            vector: npt.NDArray[np.complex128] = self.state.open_tensor(node)
            probs = (np.abs(vector) ** 2).astype(np.float64)
            probs /= np.sum(probs)
            result: Outcome = self.branch_selector.measure(node, lambda: probs[0], rng=rng)
            self.results[node] = result
            buffer = 1 / probs[result] ** 0.5
        else:
            result = self.branch_selector.measure(node, lambda: 0.5, rng=rng)
            self.results[node] = result
            buffer = 2**0.5
        if isinstance(measurement.angle, Expression):
            raise TypeError("Parameterized pattern unsupported.")
        vec = PlanarState(measurement.plane, measurement.angle).to_statevector()
        if result:
            vec = Ops.from_axis(measurement.plane.orth) @ vec
        proj_vec = vec * buffer
        self.state.measure_single(node, basis=proj_vec, rng=rng)
        return result

    @override
    def correct_byproduct(self, cmd: command.X | command.Z) -> None:
        """Perform byproduct correction.

        Parameters
        ----------
        cmd : list
            Byproduct command
            i.e. ['X' or 'Z', node, signal_domain]
        measure_method : MeasureMethod
            The measure method to use
        """
        op = Ops.X if isinstance(cmd, command.X) else Ops.Z
        self.state.evolve_single(cmd.node, op, str(cmd.kind))

    @override
    def apply_clifford(self, node: int, clifford: Clifford) -> None:
        """Apply single-qubit Clifford gate.

        Parameters
        ----------
        cmd : list
            clifford command.
            See https://arxiv.org/pdf/2212.11975.pdf for the detail.
        """
        self.state.evolve_single(node, clifford.matrix)

    @override
    def finalize(self, output_nodes: Iterable[int]) -> None:
        """Do nothing."""


def gen_str() -> str:
    """Generate dummy string for einsum."""
    return qtn.rand_uuid()


def outer_product(vectors: Sequence[npt.NDArray[np.complex128]]) -> npt.NDArray[np.complex128]:
    """Return the outer product of the given vectors.

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
    return np.array(np.einsum(subscripts, *vectors), dtype=np.complex128)
