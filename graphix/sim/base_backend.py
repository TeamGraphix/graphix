from __future__ import annotations

import numpy as np

import graphix.clifford
import graphix.command
import graphix.pauli
from graphix.command import MeasureUpdate


def op_mat_from_result(vec: tuple[float, float, float], result: bool) -> np.ndarray:
    op_mat = np.eye(2, dtype=np.complex128) / 2
    sign = (-1) ** result
    for i in range(3):
        op_mat += sign * vec[i] * graphix.clifford.CLIFFORD[i + 1] / 2
    return op_mat


def perform_measure(
    qubit: int, plane: graphix.pauli.Plane, angle: float, state, rng, pr_calc: bool = True
) -> np.ndarray:
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
    def __init__(self, pr_calc: bool = True, rng: np.random.Generator | None = None):
        """
        Parameters
        ----------
            pr_calc : bool
                whether or not to compute the probability distribution before choosing the measurement result.
                if False, measurements yield results 0/1 with 50% probabilities each.
        """
        # whether to compute the probability
        if rng is None:
            self.__rng = np.random.default_rng()
        else:
            self.__rng = rng
        self.pr_calc = pr_calc

    def _perform_measure(self, cmd: graphix.command.M):
        s_signal = np.sum([self.results[j] for j in cmd.s_domain])
        t_signal = np.sum([self.results[j] for j in cmd.t_domain])
        angle = cmd.angle * np.pi
        measure_update = MeasureUpdate.compute(cmd.plane, s_signal % 2 == 1, t_signal % 2 == 1, graphix.clifford.I)
        angle = angle * measure_update.coeff + measure_update.add_term
        loc = self.node_index.index(cmd.node)
        result = perform_measure(loc, measure_update.new_plane, angle, self.state, self.__rng, self.pr_calc)
        self.results[cmd.node] = result
        self.node_index.remove(cmd.node)
        return loc
