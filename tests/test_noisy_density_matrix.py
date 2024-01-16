import unittest

import numpy as np

from graphix import Circuit
from graphix.noise_models.noise_model import NoiseModel
from graphix.channels import (
    KrausChannel,
    depolarising_channel,
    two_qubit_depolarising_tensor_channel,
    two_qubit_depolarising_channel,
)
from graphix.noise_models.noiseless_noise_model import NoiselessNoiseModel
from graphix.ops import Ops


class TestNoiseModel(NoiseModel):
    """Test noise model for testing.
    Only return the identity channel.

    :param NoiseModel: Parent abstract class class:`graphix.noise_model.NoiseModel`
    :type NoiseModel: class
    """

    def __init__(
        self,
        prepare_error_prob=0.0,
        x_error_prob=0.0,
        z_error_prob=0.0,
        entanglement_error_prob=0.0,
        measure_channel_prob=0.0,
        measure_error_prob=0.0,
    ):
        self.prepare_error_prob = prepare_error_prob
        self.x_error_prob = x_error_prob
        self.z_error_prob = z_error_prob
        self.entanglement_error_prob = entanglement_error_prob
        self.measure_error_prob = measure_error_prob
        self.measure_channel_prob = measure_channel_prob

    def prepare_qubit(self):
        """return the channel to apply after clean single-qubit preparation. Here just identity."""
        return depolarising_channel(self.prepare_error_prob)

    def entangle(self):
        """return noise model to qubits that happens after the CZ gate"""
        # return two_qubit_depolarising_tensor_channel(self.entanglement_error_prob)
        return two_qubit_depolarising_channel(self.entanglement_error_prob)

    def measure(self):
        """apply noise to qubit to be measured."""
        return depolarising_channel(self.measure_channel_prob)

    def confuse_result(self, cmd):
        """assign wrong measurement result
        cmd = "M"
        """

        if np.random.rand() < self.measure_error_prob:
            self.simulator.results[cmd[1]] = 1 - self.simulator.results[cmd[1]]

    def byproduct_x(self):
        """apply noise to qubits after X gate correction"""
        return depolarising_channel(self.x_error_prob)

    def byproduct_z(self):
        """apply noise to qubits after Z gate correction"""
        return depolarising_channel(self.z_error_prob)

    def clifford(self):
        """apply noise to qubits that happens in the Clifford gate process"""
        # TODO list separate different Cliffords to allow customization
        return KrausChannel([{"coef": 1.0, "operator": np.eye(2)}])

    def tick_clock(self):
        """notion of time in real devices - this is where we apply effect of T1 and T2.
        we assume commands that lie between 'T' commands run simultaneously on the device.
        """
        pass


class NoisyDensityMatrixBackendTest(unittest.TestCase):
    """Test for Noisy DensityMatrixBackend simultation."""

    def setUp(self):
        # set up the random numbers
        self.rng = np.random.default_rng()  # seed=422
        np.random.seed()  # 42 vs 422

        # Hadamard pattern
        ncirc = Circuit(1)
        ncirc.h(0)
        self.hadamardpattern = ncirc.transpile()

        # Rz(alpha) pattern
        self.alpha = 2 * np.pi * self.rng.random()
        print(f"alpha is {self.alpha}")
        circ = Circuit(1)
        circ.rz(0, self.alpha)
        self.rzpattern = circ.transpile()
        self.rz_exact_res = 0.5 * np.array([[1.0, np.exp(-1j * self.alpha)], [np.exp(1j * self.alpha), 1.0]])

    # test noiseless noisy vs noiseless
    def test_noiseless_noisy_hadamard(self):

        noiselessres = self.hadamardpattern.simulate_pattern(backend="densitymatrix")
        # noiseless noise model
        noisynoiselessres = self.hadamardpattern.simulate_pattern(
            backend="densitymatrix", noise_model=NoiselessNoiseModel()
        )
        np.testing.assert_allclose(noiselessres.rho, np.array([[1.0, 0.0], [0.0, 0.0]]))
        # result should be |0>
        np.testing.assert_allclose(noisynoiselessres.rho, np.array([[1.0, 0.0], [0.0, 0.0]]))

    # test measurement confuse outcome
    def test_noisy_measure_confuse_hadamard(self):

        res = self.hadamardpattern.simulate_pattern(
            backend="densitymatrix", noise_model=TestNoiseModel(measure_error_prob=1.0)
        )
        # result should be |1>
        np.testing.assert_allclose(res.rho, np.array([[0.0, 0.0], [0.0, 1.0]]))

        # arbitrary probability
        measure_error_pr = self.rng.random()
        print(f"measure_error_pr = {measure_error_pr}")
        res = self.hadamardpattern.simulate_pattern(
            backend="densitymatrix", noise_model=TestNoiseModel(measure_error_prob=measure_error_pr)
        )
        # result should be |1>
        assert np.allclose(res.rho, np.array([[1.0, 0.0], [0.0, 0.0]])) or np.allclose(
            res.rho, np.array([[0.0, 0.0], [0.0, 1.0]])
        )

    def test_noisy_measure_channel_hadamard(self):

        measure_channel_pr = self.rng.random()
        print(f"measure_channel_pr = {measure_channel_pr}")
        # measurement error only
        res = self.hadamardpattern.simulate_pattern(
            backend="densitymatrix", noise_model=TestNoiseModel(measure_channel_prob=measure_channel_pr)
        )
        # just TP the depolarizing channel
        np.testing.assert_allclose(
            res.rho, np.array([[1 - 2 * measure_channel_pr / 3.0, 0.0], [0.0, 2 * measure_channel_pr / 3.0]])
        )

    # test Pauli X error
    def test_noisy_X_hadamard(self):

        # x error only
        x_error_pr = self.rng.random()
        print(f"x_error_pr = {x_error_pr}")
        res = self.hadamardpattern.simulate_pattern(
            backend="densitymatrix", noise_model=TestNoiseModel(x_error_prob=x_error_pr)
        )
        # analytical result since deterministic pattern output is |0>.
        # if no X applied, no noise. If X applied X noise on |0><0|

        assert np.allclose(res.rho, np.array([[1.0, 0.0], [0.0, 0.0]])) or np.allclose(
            res.rho, np.array([[1 - 2 * x_error_pr / 3.0, 0.0], [0.0, 2 * x_error_pr / 3.0]])
        )

    # test entanglement error
    def test_noisy_entanglement_hadamard(self):

        entanglement_error_pr = np.random.rand()
        res = self.hadamardpattern.simulate_pattern(
            backend="densitymatrix", noise_model=TestNoiseModel(entanglement_error_prob=entanglement_error_pr)
        )
        # analytical result for tensor depolarizing channel
        # np.testing.assert_allclose(
        #     res.rho,
        #     np.array(
        #         [
        #             [1 - 4 * entanglement_error_pr / 3.0 + 8 * entanglement_error_pr**2 / 9, 0.0],
        #             [0.0, 4 * entanglement_error_pr / 3.0 - 8 * entanglement_error_pr**2 / 9],
        #         ]
        #     ),
        # )

        # analytical result for true 2-qubit depolarizing channel
        np.testing.assert_allclose(
            res.rho,
            np.array(
                [
                    [1 - 8 * entanglement_error_pr / 15.0, 0.0],
                    [0.0, 8 * entanglement_error_pr / 15.0],
                ]
            ),
        )

    # test preparation error
    def test_noisy_preparation_hadamard(self):

        prepare_error_pr = self.rng.random()
        print(f"prepare_error_pr = {prepare_error_pr}")
        res = self.hadamardpattern.simulate_pattern(
            backend="densitymatrix", noise_model=TestNoiseModel(prepare_error_prob=prepare_error_pr)
        )
        # analytical result
        np.testing.assert_allclose(
            res.rho, np.array([[1 - 2 * prepare_error_pr / 3.0, 0.0], [0.0, 2 * prepare_error_pr / 3.0]])
        )

    ###  Test rz gate

    # test noiseless noisy vs noiseless
    def test_noiseless_noisy_rz(self):

        noiselessres = self.rzpattern.simulate_pattern(backend="densitymatrix")
        # noiseless noise model or TestNoiseModel() since all probas are 0
        noisynoiselessres = self.rzpattern.simulate_pattern(
            backend="densitymatrix", noise_model=TestNoiseModel()
        )  # NoiselessNoiseModel()
        np.testing.assert_allclose(
            noiselessres.rho, 0.5 * np.array([[1.0, np.exp(-1j * self.alpha)], [np.exp(1j * self.alpha), 1.0]])
        )
        # result should be |0>
        np.testing.assert_allclose(
            noisynoiselessres.rho, 0.5 * np.array([[1.0, np.exp(-1j * self.alpha)], [np.exp(1j * self.alpha), 1.0]])
        )

    # test preparation error
    def test_noisy_preparation_rz(self):

        prepare_error_pr = self.rng.random()
        print(f"prepare_error_pr = {prepare_error_pr}")
        res = self.rzpattern.simulate_pattern(
            backend="densitymatrix", noise_model=TestNoiseModel(prepare_error_prob=prepare_error_pr)
        )
        # analytical result
        np.testing.assert_allclose(
            res.rho,
            0.5
            * np.array(
                [
                    [
                        1.0,
                        (3 - 4 * prepare_error_pr) ** 2
                        * (3 * np.cos(self.alpha) + 1j * (-3 + 4 * prepare_error_pr) * np.sin(self.alpha))
                        / 27,
                    ],
                    [
                        (3 - 4 * prepare_error_pr) ** 2
                        * (3 * np.cos(self.alpha) - 1j * (-3 + 4 * prepare_error_pr) * np.sin(self.alpha))
                        / 27,
                        1.0,
                    ],
                ]
            ),
        )

    # test entanglement error
    def test_noisy_entanglement_rz(self):

        entanglement_error_pr = np.random.rand()
        res = self.rzpattern.simulate_pattern(
            backend="densitymatrix", noise_model=TestNoiseModel(entanglement_error_prob=entanglement_error_pr)
        )
        # analytical result for tensor depolarizing channel
        # np.testing.assert_allclose(
        #     res.rho,
        #     0.5
        #     * np.array(
        #         [
        #             [
        #                 1.0,
        #                 (-3 + 4 * entanglement_error_pr) ** 3
        #                 * (-3 * np.cos(self.alpha) + 1j * (3 - 4 * entanglement_error_pr) * np.sin(self.alpha))
        #                 / 81,
        #             ],
        #             [
        #                 (-3 + 4 * entanglement_error_pr) ** 3
        #                 * (-3 * np.cos(self.alpha) - 1j * (3 - 4 * entanglement_error_pr) * np.sin(self.alpha))
        #                 / 81,
        #                 1.0,
        #             ],
        #         ]
        #     ),
        # )

        # analytical result for true 2-qubit depolarizing channel
        np.testing.assert_allclose(
            res.rho,
            0.5
            * np.array(
                [
                    [
                        1.0,
                        np.exp(-1j * self.alpha) * (15 - 16 * entanglement_error_pr) ** 2 / 225,
                    ],
                    [
                        np.exp(1j * self.alpha) * (15 - 16 * entanglement_error_pr) ** 2 / 225,
                        1.0,
                    ],
                ]
            ),
        )

    def test_noisy_measure_channel_rz(self):

        measure_channel_pr = self.rng.random()
        print(f"measure_channel_pr = {measure_channel_pr}")
        # measurement error only
        res = self.rzpattern.simulate_pattern(
            backend="densitymatrix", noise_model=TestNoiseModel(measure_channel_prob=measure_channel_pr)
        )

        np.testing.assert_allclose(
            res.rho,
            0.5
            * np.array(
                [
                    [
                        1.0,
                        (-3 + 4 * measure_channel_pr)
                        * (-3 * np.cos(self.alpha) + 1j * (3 - 4 * measure_channel_pr) * np.sin(self.alpha))
                        / 9,
                    ],
                    [
                        (-3 + 4 * measure_channel_pr)
                        * (-3 * np.cos(self.alpha) - 1j * (3 - 4 * measure_channel_pr) * np.sin(self.alpha))
                        / 9,
                        1.0,
                    ],
                ]
            ),
        )

    def test_noisy_X_rz(self):

        # x error only
        x_error_pr = self.rng.random()
        print(f"x_error_pr = {x_error_pr}")
        res = self.rzpattern.simulate_pattern(
            backend="densitymatrix", noise_model=TestNoiseModel(x_error_prob=x_error_pr)
        )

        # only two cases: if no X correction, Z or no Z correction but exact result.
        # If X correction the noise result is the same with or without the PERFECT Z correction.
        assert np.allclose(
            res.rho, 0.5 * np.array([[1.0, np.exp(-1j * self.alpha)], [np.exp(1j * self.alpha), 1.0]])
        ) or np.allclose(
            res.rho,
            0.5
            * np.array(
                [
                    [1.0, np.exp(-1j * self.alpha) * (3 - 4 * x_error_pr) / 3],
                    [np.exp(1j * self.alpha) * (3 - 4 * x_error_pr) / 3, 1.0],
                ]
            ),
        )

    def test_noisy_Z_rz(self):

        # z error only
        z_error_pr = self.rng.random()
        print(f"z_error_pr = {z_error_pr}")
        res = self.rzpattern.simulate_pattern(
            backend="densitymatrix", noise_model=TestNoiseModel(z_error_prob=z_error_pr)
        )

        # only two cases: if no Z correction, X or no X correction but exact result.
        # If Z correction the noise result is the same with or without the PERFECT X correction.
        assert np.allclose(
            res.rho, 0.5 * np.array([[1.0, np.exp(-1j * self.alpha)], [np.exp(1j * self.alpha), 1.0]])
        ) or np.allclose(
            res.rho,
            0.5
            * np.array(
                [
                    [1.0, np.exp(-1j * self.alpha) * (3 - 4 * z_error_pr) / 3],
                    [np.exp(1j * self.alpha) * (3 - 4 * z_error_pr) / 3, 1.0],
                ]
            ),
        )

    def test_noisy_XZ_rz(self):

        # x and z errors
        x_error_pr = self.rng.random()
        print(f"x_error_pr = {x_error_pr}")
        z_error_pr = self.rng.random()
        print(f"z_error_pr = {z_error_pr}")
        res = self.rzpattern.simulate_pattern(
            backend="densitymatrix", noise_model=TestNoiseModel(x_error_prob=x_error_pr, z_error_prob=z_error_pr)
        )

        # 4 cases : no corr, noisy X, noisy Z, noisy XZ.
        assert (
            np.allclose(res.rho, 0.5 * np.array([[1.0, np.exp(-1j * self.alpha)], [np.exp(1j * self.alpha), 1.0]]))
            or np.allclose(
                res.rho,
                0.5
                * np.array(
                    [
                        [1.0, np.exp(-1j * self.alpha) * (3 - 4 * x_error_pr) * (3 - 4 * z_error_pr) / 9],
                        [np.exp(1j * self.alpha) * (3 - 4 * x_error_pr) * (3 - 4 * z_error_pr) / 9, 1.0],
                    ]
                ),
            )
            or np.allclose(
                res.rho,
                0.5
                * np.array(
                    [
                        [1.0, np.exp(-1j * self.alpha) * (3 - 4 * z_error_pr) / 3],
                        [np.exp(1j * self.alpha) * (3 - 4 * z_error_pr) / 3, 1.0],
                    ]
                ),
            )
            or np.allclose(
                res.rho,
                0.5
                * np.array(
                    [
                        [1.0, np.exp(-1j * self.alpha) * (3 - 4 * x_error_pr) / 3],
                        [np.exp(1j * self.alpha) * (3 - 4 * x_error_pr) / 3, 1.0],
                    ]
                ),
            )
        )

    # test measurement confuse outcome
    def test_noisy_measure_confuse_rz(self):

        # probability 1 to shift both outcome
        res = self.rzpattern.simulate_pattern(
            backend="densitymatrix", noise_model=TestNoiseModel(measure_error_prob=1.0)
        )
        # result X, XZ or Z

        assert (
            np.allclose(res.rho, Ops.x @ self.rz_exact_res @ Ops.x)
            or np.allclose(res.rho, Ops.z @ self.rz_exact_res @ Ops.z)
            or np.allclose(res.rho, Ops.z @ Ops.x @ self.rz_exact_res @ Ops.x @ Ops.z)
        )

        # arbitrary probability
        measure_error_pr = self.rng.random()
        print(f"measure_error_pr = {measure_error_pr}")
        res = self.rzpattern.simulate_pattern(
            backend="densitymatrix", noise_model=TestNoiseModel(measure_error_prob=measure_error_pr)
        )
        # just add the case without readout errors
        assert (
            np.allclose(res.rho, self.rz_exact_res)
            or np.allclose(res.rho, Ops.x @ self.rz_exact_res @ Ops.x)
            or np.allclose(res.rho, Ops.z @ self.rz_exact_res @ Ops.z)
            or np.allclose(res.rho, Ops.z @ Ops.x @ self.rz_exact_res @ Ops.x @ Ops.z)
        )


if __name__ == "__main__":
    np.random.seed(32)
    unittest.main()
