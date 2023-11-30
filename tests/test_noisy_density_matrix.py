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
    # NOTE possible to declare common variables here?
    

    # test noiseless noisy vs noiseless
    def test_noiseless_noisy_hadamard(self):

        # Hadamard gate test
        ncirc = Circuit(1)
        ncirc.h(0)
        hadamardpattern = ncirc.transpile()
        # no noise
        noiselessres = hadamardpattern.simulate_pattern(backend='densitymatrix')
        # noiseless noise model
        noisynoiselessres = hadamardpattern.simulate_pattern(backend='densitymatrix', noise_model=NoiselessNoiseModel())
        np.testing.assert_allclose(noiselessres.rho, np.array([[1., 0.],[0., 0.]]))
        # result should be |0>
        np.testing.assert_allclose(noisynoiselessres.rho, np.array([[1., 0.],[0., 0.]]))

    # test measurement confuse outcome
    def test_noisy_measure_confuse_hadamard(self):

        # Hadamard gate test
        ncirc = Circuit(1)
        ncirc.h(0)
        hadamardpattern = ncirc.transpile()
        # measurement error only
        res = hadamardpattern.simulate_pattern(backend='densitymatrix', noise_model=TestNoiseModel(measure_error_prob=1.))
        # result should be |1>
        np.testing.assert_allclose(res.rho, np.array([[0., 0.],[0., 1.]]))

    def test_noisy_measure_channel_hadamard(self):

        # Hadamard gate test
        ncirc = Circuit(1)
        ncirc.h(0)
        hadamardpattern = ncirc.transpile()
        # measurement error only
        res = hadamardpattern.simulate_pattern(backend='densitymatrix', noise_model=TestNoiseModel(measure_channel_prob=1.))
        # result should be |1>
        np.testing.assert_allclose(res.rho, np.array([[0., 0.],[0., 1.]]))

    # test Pauli X error
    # error = 0.75 gives maximally mixed Id/2
    def test_noisy_X_hadamard(self):

        # Hadamard gate test
        ncirc = Circuit(1)
        ncirc.h(0)
        hadamardpattern = ncirc.transpile()
        # x error only
        x_error_pr = np.random.rand()
        res = hadamardpattern.simulate_pattern(backend='densitymatrix', noise_model=TestNoiseModel(x_error_prob=x_error_pr))                          
        np.testing.assert_allclose(res.rho, np.array([[1-2*x_error_pr/3., 0.],[0., 2*x_error_pr/3.]]))

    # test entanglement error
    def test_noisy_entanglement_hadamard(self):

        # Hadamard gate test
        ncirc = Circuit(1)
        ncirc.h(0)
        hadamardpattern = ncirc.transpile()
        # x error only
        entanglement_error_pr = np.random.rand()
        res = hadamardpattern.simulate_pattern(backend='densitymatrix', noise_model=TestNoiseModel(entanglement_error_prob=entanglement_error_pr))                          
        np.testing.assert_allclose(res.rho, np.array([[1-2*entanglement_error_pr/3., 0.],[0., 2*entanglement_error_pr/3.]]))



       

    # def test_dephase(self):
    #     def run(p, pattern, max_qubit_num=12):
    #         backend = DensityMatrixBackend(pattern, max_qubit_num=max_qubit_num)
    #         for cmd in pattern.seq:
    #             if cmd[0] == "N":
    #                 backend.add_nodes([cmd[1]])
    #             elif cmd[0] == "E":
    #                 backend.entangle_nodes(cmd[1])
    #                 backend.dephase(p)
    #             elif cmd[0] == "M":
    #                 backend.measure(cmd)
    #                 backend.dephase(p)
    #             elif cmd[0] == "X":
    #                 backend.correct_byproduct(cmd)
    #                 backend.dephase(p)
    #             elif cmd[0] == "Z":
    #                 backend.correct_byproduct(cmd)
    #                 backend.dephase(p)
    #             elif cmd[0] == "C":
    #                 backend.apply_clifford(cmd)
    #                 backend.dephase(p)
    #             elif cmd[0] == "T":
    #                 backend.dephase(p)
    #             else:
    #                 raise ValueError("invalid commands")
    #             if pattern.seq[-1] == cmd:
    #                 backend.finalize()
    #         return backend

    #     # Test for Rx(pi/4)
    #     circ = Circuit(1)
    #     circ.rx(0, np.pi / 4)
    #     pattern = circ.transpile()
    #     backend1 = run(0, pattern)
    #     backend2 = run(1, pattern)
    #     np.testing.assert_allclose(backend1.state.rho, backend2.state.rho)

    #     # Test for Rz(pi/3)
    #     circ = Circuit(1)
    #     circ.rz(0, np.pi / 3)
    #     pattern = circ.transpile()
    #     dm_backend = run(1, pattern)
    #     sv_backend = StatevectorBackend(pattern)
    #     sv_backend.add_nodes([0, 1, 2])
    #     sv_backend.entangle_nodes((0, 1))
    #     sv_backend.entangle_nodes((1, 2))
    #     sv_backend.measure(pattern.seq[-4])
    #     sv_backend.measure(pattern.seq[-3])
    #     sv_backend.correct_byproduct(pattern.seq[-2])
    #     sv_backend.correct_byproduct(pattern.seq[-1])
    #     sv_backend.finalize()
    #     np.testing.assert_allclose(dm_backend.state.fidelity(sv_backend.state.psi), 0.25)

    #     # Test for 3-qubit QFT
    #     def cp(circuit, theta, control, target):
    #         """Controlled rotation gate, decomposed"""
    #         circuit.rz(control, theta / 2)
    #         circuit.rz(target, theta / 2)
    #         circuit.cnot(control, target)
    #         circuit.rz(target, -1 * theta / 2)
    #         circuit.cnot(control, target)

    #     def swap(circuit, a, b):
    #         """swap gate, decomposed"""
    #         circuit.cnot(a, b)
    #         circuit.cnot(b, a)
    #         circuit.cnot(a, b)

    #     def qft_circ():
    #         circ = Circuit(3)
    #         for i in range(3):
    #             circ.h(i)
    #         circ.x(1)
    #         circ.x(2)

    #         circ.h(2)
    #         cp(circ, np.pi / 4, 0, 2)
    #         cp(circ, np.pi / 2, 1, 2)
    #         circ.h(1)
    #         cp(circ, np.pi / 2, 0, 1)
    #         circ.h(0)
    #         swap(circ, 0, 2)
    #         return circ

    #     # no-noise case
    #     circ = qft_circ()
    #     pattern = circ.transpile()
    #     dm_backend = run(0, pattern)
    #     state = circ.simulate_statevector().flatten()
    #     np.testing.assert_allclose(dm_backend.state.fidelity(state), 1)

    #     # noisy case vs exact 3-qubit QFT result
    #     circ = qft_circ()
    #     pattern = circ.transpile()
    #     p = np.random.rand() * 0 + 0.8
    #     dm_backend = run(p, pattern)
    #     noisy_state = circ.simulate_statevector().flatten()

    #     sv = Statevec(nqubit=3)
    #     omega = np.exp(2j * np.pi / 8)
    #     qft_matrix = np.array(
    #         [
    #             [1, 1, 1, 1, 1, 1, 1, 1],
    #             [1, omega, omega**2, omega**3, omega**4, omega**5, omega**6, omega**7],
    #             [1, omega**2, omega**4, omega**6, 1, omega**2, omega**4, omega**6],
    #             [1, omega**3, omega**6, omega, omega**4, omega**7, omega**2, omega**5],
    #             [1, omega**4, 1, omega**4, 1, omega**4, 1, omega**4],
    #             [1, omega**5, omega**2, omega**7, omega**4, omega, omega**6, omega**3],
    #             [1, omega**6, omega**4, omega**2, 1, omega**6, omega**4, omega**2],
    #             [1, omega**7, omega**6, omega**5, omega**4, omega**3, omega**2, omega],
    #         ]
    #     ) / np.sqrt(8)
    #     exact_qft_state = qft_matrix @ sv.psi.flatten()
    #     np.testing.assert_allclose(dm_backend.state.fidelity(noisy_state), dm_backend.state.fidelity(exact_qft_state))


if __name__ == "__main__":
    np.random.seed(32)
    unittest.main()
