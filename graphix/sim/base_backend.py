import abc
import dataclasses
import numpy as np
import graphix.clifford
import graphix.pauli


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

    def _perform_measure(self, node, measurement_description):
        # measurement_description = self.__measure_method.get_measurement_description(cmd, self.results)
        vec = measurement_description.plane.polar(measurement_description.angle)
        loc = self.node_index.index(node)

        def op_mat_from_result(result: bool) -> np.ndarray:
            op_mat = np.eye(2, dtype=np.complex128) / 2
            sign = (-1) ** result
            for i in range(3):
                op_mat += sign * vec[i] * graphix.clifford.CLIFFORD[i + 1] / 2
            return op_mat

        if self.pr_calc:
            op_mat = op_mat_from_result(False)
            prob_0 = self.state.expectation_single(op_mat, loc)
            result = np.random.rand() > prob_0
            if result:
                op_mat = op_mat_from_result(True)
        else:
            # choose the measurement result randomly
            result = np.random.choice([0, 1])
            op_mat = op_mat_from_result(result)
        self.results[node] = result
        # self.__measure_method.set_measure_result(node, result)
        self.state.evolve_single(op_mat, loc)
        self.node_index.remove(node)
        return loc, result
