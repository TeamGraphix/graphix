from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import graphix.clifford
import graphix.command
import graphix.pauli
import graphix.states
from graphix.clifford import Clifford
from graphix.ops import Ops
from graphix.pauli import Plane


@dataclass
class MeasurementDescription:
    plane: Plane
    angle: float


class NodeIndex:
    def __init__(self, d: dict[int, int] = None) -> None:
        if d is None:
            d = dict()
        self.__dict = d

    def __getitem__(self, index: int) -> int:
        return self.__dict[index]

    def __iter__(self) -> collections.abc.Iterator[int]:
        return iter(self.__dict.keys())

    def extend(self, nodes: collections.abc.Iterable[int]) -> NodeIndex:
        d = self.__dict.copy()
        base = len(d)
        for index, node in enumerate(nodes):
            d[node] = base + index
        return NodeIndex(d)

    def remove(self, node: int) -> NodeIndex:
        index = self.__dict[node]
        d = {k: v - 1 if v > index else v for k, v in self.__dict.items() if k != node}
        return NodeIndex(d)

    def swap(self, i: int, j: int) -> NodeIndex:
        d = {k: j if v == i else i if v == j else v for k, v in self.__dict.items()}
        return NodeIndex(d)

    def to_list(self) -> list[int]:
        result = [0] * len(self.__dict)
        for node, index in self.__dict.items():
            result[index] = node
        return result


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
        result = rng.choice([False, True])
        op_mat = op_mat_from_result(vec, result)
    state.evolve_single(op_mat, qubit)
    return result


class Backend:
    def __init__(self, state: State, pr_calc: bool = True, rng: np.random.Generator = None):
        """
        Parameters
        ----------
            pr_calc : bool
                whether or not to compute the probability distribution before choosing the measurement result.
                if False, measurements yield results 0/1 with 50% probabilities each.
        """
        self.state = state
        self.node_index = NodeIndex()
        if not isinstance(pr_calc, bool):
            raise TypeError("`pr_calc` should be bool")
        # whether to compute the probability
        self.__pr_calc = pr_calc
        if rng is None:
            rng = np.random.default_rng()
        self.__rng = rng

    def with_changes(self, *, state: State | None = None, node_index: NodeIndex | None = None) -> Backend:
        result = self.__new__(self.__class__)
        if state is None:
            result.state = self.state
        else:
            result.state = state
        if node_index is None:
            result.node_index = self.node_index
        else:
            result.node_index = node_index
        result.__pr_calc = self.__pr_calc
        result.__rng = self.__rng
        return result

    def add_nodes(self, nodes, data=graphix.states.BasicStates.PLUS) -> Backend:
        """add new qubit(s) to statevector in argument
        and assign the corresponding node number
        to list self.node_index.

        Parameters
        ----------
        nodes : list of node indices
        """
        new_state = self.state.copy()
        new_state.add_nodes(nqubit=len(nodes), data=data)
        new_node_index = self.node_index.extend(nodes)
        return self.with_changes(state=new_state, node_index=new_node_index)

    def entangle_nodes(self, edge: tuple[int, int]) -> Backend:
        """Apply CZ gate to two connected nodes

        Parameters
        ----------
        edge : tuple (i, j)
            a pair of node indices
        """
        target = self.node_index[edge[0]]
        control = self.node_index[edge[1]]
        new_state = self.state.copy()
        new_state.entangle((target, control))
        return self.with_changes(state=new_state)

    def measure(self, node: int, measurement_description: MeasurementDescription) -> Backend:
        """Perform measurement of a node in the internal statevector and trace out the qubit

        Parameters
        ----------
        node: int
        measurement_description: MeasurementDescription
        """
        loc = self.node_index[node]
        new_state = self.state.copy()
        result = perform_measure(
            loc, measurement_description.plane, measurement_description.angle, new_state, self.__rng, self.__pr_calc
        )
        new_node_index = self.node_index.remove(node)
        new_state.remove_qubit(loc)
        return self.with_changes(state=new_state, node_index=new_node_index), result

    def correct_byproduct(self, cmd, measure_method) -> Backend:
        """Byproduct correction
        correct for the X or Z byproduct operators,
        by applying the X or Z gate.
        """
        if np.mod(np.sum([measure_method.get_measure_result(j) for j in cmd[2]]), 2) == 1:
            if cmd[0] == "X":
                op = Ops.x
            elif cmd[0] == "Z":
                op = Ops.z
            return self.apply_single(node=cmd[1], op=op)
        else:
            return self

    def apply_single(self, node, op) -> Backend:
        index = self.node_index[node]
        new_state = self.state.copy()
        new_state.evolve_single(op=op, i=index)
        return self.with_changes(state=new_state)

    def apply_clifford(self, clifford: Clifford) -> Backend:
        """Apply single-qubit Clifford gate,
        specified by vop index specified in graphix.clifford.CLIFFORD
        """
        loc = self.node_index[cmd[1]]
        new_state = self.state.copy()
        new_state.evolve_single(clifford.matrix, loc)
        return self.with_changes(state=new_state)

    def sort_qubits(self, output_nodes) -> Backend:
        """sort the qubit order in internal statevector"""
        new_state = self.state
        new_node_index = self.node_index
        for i, ind in enumerate(output_nodes):
            if new_node_index[ind] != i:
                move_from = new_node_index[ind]
                new_state = new_state.copy()
                new_state.swap((i, move_from))
                new_node_index = new_node_index.swap(i, move_from)
        return self.with_changes(state=new_state, node_index=new_node_index)

    def finalize(self, output_nodes) -> Backend:
        """to be run at the end of pattern simulation."""
        return self.sort_qubits(output_nodes)

    @property
    def Nqubit(self) -> int:
        return self.state.Nqubit
