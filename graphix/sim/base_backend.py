import typing

import numpy as np

import graphix.clifford
import graphix.pauli


def op_mat_from_result(vec: typing.Tuple[float, float, float], result: bool) -> np.ndarray:
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
        result = rng.rand() > prob_0
        if result:
            op_mat = op_mat_from_result(vec, True)
    else:
        # choose the measurement result randomly
        result = np.random.choice([0, 1])
        op_mat = op_mat_from_result(vec, result)
    state.evolve_single(op_mat, qubit)
    return result


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

    def _perform_measure(self, cmd):
        s_signal = np.sum([self.results[j] for j in cmd[4]])
        t_signal = np.sum([self.results[j] for j in cmd[5]])
        angle = cmd[3] * np.pi
        if len(cmd) == 7:
            vop = cmd[6]
        else:
            vop = 0
        measure_update = graphix.pauli.MeasureUpdate.compute(
            graphix.pauli.Plane[cmd[2]], s_signal % 2 == 1, t_signal % 2 == 1, graphix.clifford.TABLE[vop]
        )
        angle = angle * measure_update.coeff + measure_update.add_term
        loc = self.node_index.index(cmd[1])
        result = perform_measure(loc, measure_update.new_plane, angle, self.state, np.random, self.pr_calc)
        self.results[cmd[1]] = result
        self.node_index.remove(cmd[1])
        return loc
