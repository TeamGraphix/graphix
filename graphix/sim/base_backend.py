from __future__ import annotations

import abc
import collections
import dataclasses

import numpy as np

import graphix.clifford
import graphix.pauli
import graphix.states


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


@dataclasses.dataclass
class IndexedState:
    state: State
    node_index: NodeIndex


class Backend:
    def __init__(self, pr_calc: bool = True):
        """
        Parameters
        ----------
            pr_calc : bool
                whether or not to compute the probability distribution before choosing the measurement result.
                if False, measurements yield results 0/1 with 50% probabilities each.
        """
        # whether to compute the probability
        self.pr_calc = pr_calc

    def _perform_measure(self, state: IndexedState, node, measurement_description):
        # measurement_description = self.__measure_method.get_measurement_description(cmd, self.results)
        vec = measurement_description.plane.polar(measurement_description.angle)
        loc = state.node_index[node]

        def op_mat_from_result(result: bool) -> np.ndarray:
            op_mat = np.eye(2, dtype=np.complex128) / 2
            sign = (-1) ** result
            for i in range(3):
                op_mat += sign * vec[i] * graphix.clifford.CLIFFORD[i + 1] / 2
            return op_mat

        if self.pr_calc:
            op_mat = op_mat_from_result(False)
            prob_0 = state.state.expectation_single(op_mat, loc)
            result = np.random.rand() > prob_0
            if result:
                op_mat = op_mat_from_result(True)
        else:
            # choose the measurement result randomly
            result = np.random.choice([0, 1])
            op_mat = op_mat_from_result(result)
        # self.__measure_method.set_measure_result(node, result)
        new_state = state.state.evolve_single(op_mat, loc)
        new_node_index = state.node_index.remove(node)
        return IndexedState(state=new_state, node_index=new_node_index), loc, result

    def initial_state(self) -> IndexedState: ...

    def sort_qubits(self, state: IndexedState, output_nodes) -> IndexedState:
        """sort the qubit order in internal statevector"""
        new_state = state.state
        new_node_index = state.node_index
        for i, ind in enumerate(output_nodes):
            if new_node_index[ind] != i:
                move_from = new_node_index[ind]
                new_state = new_state.swap((i, move_from))
                new_node_index = new_node_index.swap(i, move_from)
        return IndexedState(new_state, new_node_index)

    def finalize(self, state: IndexedState, output_nodes) -> IndexedState:
        """to be run at the end of pattern simulation."""
        return self.sort_qubits(state, output_nodes)
