import random
import unittest
from copy import deepcopy

import numpy as np


from graphix import Circuit
from graphix.channels import Channel, create_dephasing_channel, create_depolarising_channel
from graphix.ops import Ops
from graphix.sim.density_matrix import DensityMatrix, DensityMatrixBackend
from graphix.sim.statevec import CNOT_TENSOR, CZ_TENSOR, SWAP_TENSOR, Statevec, StatevectorBackend
import tests.random_objects as randobj
from graphix.noise_models.noiseless_noise_model import NoiselessNoiseModel
from graphix.noise_models.test_noise_model import TestNoiseModel


class NoisyDensityMatrixBackendTest(unittest.TestCase):
    """Test for Noisy DensityMatrixBackend simultation."""

    def setUp(self):
        # set up the random numbers
        self.rng = np.random.default_rng()#seed=422
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



    # test noiseless noisy vs noiseless
    def test_noiseless_noisy_hadamard(self):

        noiselessres = self.hadamardpattern.simulate_pattern(backend="densitymatrix")
        # noiseless noise model
        noisynoiselessres = self.hadamardpattern.simulate_pattern(backend="densitymatrix", noise_model=NoiselessNoiseModel())
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
    # error = 0.75 gives maximally mixed Id/2
    def test_noisy_X_hadamard(self):

        # x error only
        x_error_pr = self.rng.random()
        print(f"x_error_pr = {x_error_pr}")
        res = self.hadamardpattern.simulate_pattern(
            backend="densitymatrix", noise_model=TestNoiseModel(x_error_prob=x_error_pr)
        )
        # analytical result since deterministic pattern output is |0>.
        # if no X applied, no noise. If x applied x noise on |0><0|

        print("state \n", res.rho)

        assert np.allclose(res.rho, np.array([[1.0, 0.0], [0.0, 0.0]])) or np.allclose(
            res.rho, np.array([[1 - 2 * x_error_pr / 3.0, 0.0], [0.0, 2 * x_error_pr / 3.0]])
        )

    # test entanglement error
    def test_noisy_entanglement_hadamard(self):

        entanglement_error_pr = np.random.rand()
        res = self.hadamardpattern.simulate_pattern(
            backend="densitymatrix", noise_model=TestNoiseModel(entanglement_error_prob=entanglement_error_pr)
        )
        # analytical result
        np.testing.assert_allclose(
            res.rho,
            np.array(
                [
                    [1 - 4 * entanglement_error_pr / 3.0 + 8 * entanglement_error_pr**2 / 9, 0.0],
                    [0.0, 4 * entanglement_error_pr / 3.0 - 8 * entanglement_error_pr**2 / 9],
                ]
            ),
        )

    # test preparation error
    def test_noisy_preparation_hadamard(self):
        
        prepare_error_pr =  self.rng.random()
        print(f"prepare_error_pr = {prepare_error_pr}")
        res = self.hadamardpattern.simulate_pattern(
            backend="densitymatrix", noise_model=TestNoiseModel(prepare_error_prob=prepare_error_pr)
        )
        # analytical result
        np.testing.assert_allclose(
            res.rho,
            np.array([[1 - 2 * prepare_error_pr / 3.0, 0.0], [0.0, 2 * prepare_error_pr / 3.0]])
        )

    ###  now test rz gate

    # test noiseless noisy vs noiseless
    def test_noiseless_noisy_rz(self):

        noiselessres = self.rzpattern.simulate_pattern(backend="densitymatrix")
        # noiseless noise model
        noisynoiselessres = self.rzpattern.simulate_pattern(backend="densitymatrix", noise_model=NoiselessNoiseModel())
        np.testing.assert_allclose(noiselessres.rho, 0.5 * np.array([[1., np.exp(-1j*self.alpha)], [np.exp(1j*self.alpha), 1.]]))
        # result should be |0>
        np.testing.assert_allclose(noisynoiselessres.rho, 0.5 * np.array([[1., np.exp(-1j*self.alpha)], [np.exp(1j*self.alpha), 1.]]))


     # test preparation error
    def test_noisy_preparation_rz(self):
        
        prepare_error_pr =  self.rng.random()
        print(f"prepare_error_pr = {prepare_error_pr}")
        res = self.rzpattern.simulate_pattern(
            backend="densitymatrix", noise_model=TestNoiseModel(prepare_error_prob=prepare_error_pr)
        )
        # analytical result
        np.testing.assert_allclose(
            res.rho,
            0.5 * np.array([[1., (3 - 4 * prepare_error_pr)**2*(3 * np.cos(self.alpha) + 1j*(-3 + 4*prepare_error_pr)*np.sin(self.alpha))/27 ], [(3 - 4 * prepare_error_pr)**2*(3 * np.cos(self.alpha) - 1j*(-3 + 4*prepare_error_pr)*np.sin(self.alpha))/27 ,1.]])
        )

    # test entanglement error
    def test_noisy_entanglement_rz(self):

        entanglement_error_pr = np.random.rand()
        res = self.rzpattern.simulate_pattern(
            backend="densitymatrix", noise_model=TestNoiseModel(entanglement_error_prob=entanglement_error_pr)
        )
        # analytical result
        np.testing.assert_allclose(
            res.rho,
            0.5*np.array(
                [
                    [1., (-3 + 4*entanglement_error_pr)**3 * (-3*np.cos(self.alpha) + 1j*(3 - 4*entanglement_error_pr)*np.sin(self.alpha))/81],
                    [(-3 + 4*entanglement_error_pr)**3 * (-3*np.cos(self.alpha) - 1j*(3 - 4*entanglement_error_pr)*np.sin(self.alpha))/81, 1.],
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
        # just TP the depolarizing channel
        np.testing.assert_allclose(
            res.rho,
            0.5 * np.array([[1., (-3 + 4 * measure_channel_pr)*(-3 * np.cos(self.alpha) + 1j*(3 - 4 * measure_channel_pr)*np.sin(self.alpha))/9 ], [(-3 + 4 * measure_channel_pr)*(-3 * np.cos(self.alpha) - 1j*(3 - 4 * measure_channel_pr)*np.sin(self.alpha))/9 ,1.]])
        )
        print("SUCCESSSSSS!!!!!!!!")

# continue with X, Z corrections!
# investigate determinism and proba 1/2


# NOTE useless if we use pytest
if __name__ == "__main__":
    np.random.seed(32)
    unittest.main()
