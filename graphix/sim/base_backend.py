from __future__ import annotations

import typing
from dataclasses import dataclass

import numpy as np

import graphix.clifford
import graphix.pauli
import graphix.states
from graphix.command import CommandKind
from graphix.ops import Ops

if typing.TYPE_CHECKING:
    import collections

    from graphix.pauli import Plane


@dataclass
class MeasurementDescription:
    plane: Plane
    angle: float


class NodeIndex:
    def __init__(self) -> None:
        self.__dict = dict()
        self.__list = []

    def copy(self) -> NodeIndex:
        result = self.__new__(self.__class__)
        result.__dict = self.__dict.copy()
        result.__list = self.__list.copy()
        return result

    def __getitem__(self, index: int) -> int:
        return self.__list[index]

    def index(self, node: int) -> int:
        return self.__dict[node]

    def __iter__(self) -> collections.abc.Iterator[int]:
        return iter(self.__list)

    def __len__(self) -> int:
        return len(self.__list)

    def extend(self, nodes: collections.abc.Iterable[int]) -> None:
        base = len(self)
        self.__list.extend(nodes)
        # The following loop iterates over `self.__list[base:]` instead of `nodes`
        # because the iterable `nodes` can be transient and consumed by the
        # `self.__list.extend` on the line just above.
        for index, node in enumerate(self.__list[base:]):
            self.__dict[node] = base + index

    def remove(self, node: int) -> None:
        index = self.__dict[node]
        del self.__list[index]
        for new_index, node in enumerate(self.__list[index:], start=index):
            self.__dict[node] = new_index

    def swap(self, i: int, j: int) -> None:
        node_i = self.__list[i]
        node_j = self.__list[j]
        self.__list[i] = node_j
        self.__list[j] = node_i
        self.__dict[node_i] = j
        self.__dict[node_j] = i


class State:
    pass


def op_mat_from_result(vec: tuple[float, float, float], result: bool) -> np.ndarray:
    op_mat = np.eye(2, dtype=np.complex128) / 2
    sign = (-1) ** result
    for i in range(3):
        op_mat += sign * vec[i] * graphix.clifford.CLIFFORD[i + 1] / 2
    return op_mat


def perform_measure(
    qubit: int, plane: graphix.pauli.Plane, angle: float, state, rng: np.random.Generator, pr_calc: bool = True
) -> bool:
    vec = plane.polar(angle)
    if pr_calc:
        op_mat = op_mat_from_result(vec, False)
        prob_0 = state.expectation_single(op_mat, qubit)
        result = rng.random() > prob_0
        if result:
            op_mat = op_mat_from_result(vec, True)
    else:
        # choose the measurement result randomly
        result = rng.choice([0, 1])
        op_mat = op_mat_from_result(vec, result)
    state.evolve_single(op_mat, qubit)
    return result


class Backend:
    def __init__(
        self, state: State, node_index: NodeIndex | None = None, pr_calc: bool = True, rng: np.random.Generator = None
    ):
        """
        Parameters
        ----------
            pr_calc : bool
                whether or not to compute the probability distribution before choosing the measurement result.
                if False, measurements yield results 0/1 with 50% probabilities each.
        """
        self.__state = state
        if node_index is None:
            self.__node_index = NodeIndex()
        else:
            self.__node_index = node_index.copy()
        if not isinstance(pr_calc, bool):
            raise TypeError("`pr_calc` should be bool")
        # whether to compute the probability
        self.__pr_calc = pr_calc
        if rng is None:
            rng = np.random.default_rng()
        self.__rng = rng

    def copy(self) -> Backend:
        return Backend(self.__state, self.__node_index, self.__pr_calc, self.__rng)

    @property
    def state(self) -> State:
        return self.__state

    @property
    def node_index(self) -> NodeIndex:
        return self.__node_index

    def add_nodes(self, nodes, data=graphix.states.BasicStates.PLUS) -> None:
        """add new qubit(s) to statevector in argument
        and assign the corresponding node number
        to list self.node_index.

        Parameters
        ----------
        nodes : list of node indices
        """
        self.state.add_nodes(nqubit=len(nodes), data=data)
        self.node_index.extend(nodes)

    def entangle_nodes(self, edge: tuple[int, int]) -> None:
        """Apply CZ gate to two connected nodes

        Parameters
        ----------
        edge : tuple (i, j)
            a pair of node indices
        """
        target = self.node_index.index(edge[0])
        control = self.node_index.index(edge[1])
        self.state.entangle((target, control))

    def measure(self, node: int, measurement_description: MeasurementDescription) -> bool:
        """Perform measurement of a node in the internal statevector and trace out the qubit

        Parameters
        ----------
        node: int
        measurement_description: MeasurementDescription
        """
        loc = self.node_index.index(node)
        result = perform_measure(
            loc, measurement_description.plane, measurement_description.angle, self.state, self.__rng, self.__pr_calc
        )
        self.node_index.remove(node)
        self.state.remove_qubit(loc)
        return result

    def correct_byproduct(self, cmd, measure_method) -> None:
        """Byproduct correction
        correct for the X or Z byproduct operators,
        by applying the X or Z gate.
        """
        if np.mod(sum([measure_method.get_measure_result(j) for j in cmd.domain]), 2) == 1:
            if cmd.kind == CommandKind.X:
                op = Ops.x
            elif cmd.kind == CommandKind.Z:
                op = Ops.z
            self.apply_single(node=cmd.node, op=op)

    def apply_single(self, node, op) -> None:
        index = self.node_index.index(node)
        self.state.evolve_single(op=op, i=index)

    def apply_clifford(self, node: int, clifford: graphix.clifford.Clifford) -> None:
        """Apply single-qubit Clifford gate,
        specified by vop index specified in graphix.clifford.CLIFFORD
        """
        loc = self.node_index.index(node)
        self.state.evolve_single(clifford.matrix, loc)

    def sort_qubits(self, output_nodes) -> None:
        """sort the qubit order in internal statevector"""
        for i, ind in enumerate(output_nodes):
            if self.node_index.index(ind) != i:
                move_from = self.node_index.index(ind)
                self.state.swap((i, move_from))
                self.node_index.swap(i, move_from)

    def finalize(self, output_nodes) -> None:
        """to be run at the end of pattern simulation."""
        self.sort_qubits(output_nodes)

    @property
    def Nqubit(self) -> int:
        return self.state.Nqubit
