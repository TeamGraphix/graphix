from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest

from graphix.channels import KrausChannel, KrausData, depolarising_channel, two_qubit_depolarising_channel
from graphix.noise_models.noise_model import NoiseModel
from graphix.noise_models.noiseless_noise_model import NoiselessNoiseModel
from graphix.ops import Ops
from graphix.transpiler import Circuit

if TYPE_CHECKING:
    from numpy.random import Generator

    from graphix.pattern import Pattern


class NoiseModelTester(NoiseModel):
    """Noise model for testing.

    Only return the identity channel.

    :param NoiseModel: Parent abstract class class:`graphix.noise_model.NoiseModel`
    :type NoiseModel: class
    """

    def __init__(
        self,
        prepare_error_prob: float = 0.0,
        x_error_prob: float = 0.0,
        z_error_prob: float = 0.0,
        entanglement_error_prob: float = 0.0,
        measure_channel_prob: float = 0.0,
        measure_error_prob: float = 0.0,
    ) -> None:
        self.prepare_error_prob = prepare_error_prob
        self.x_error_prob = x_error_prob
        self.z_error_prob = z_error_prob
        self.entanglement_error_prob = entanglement_error_prob
        self.measure_error_prob = measure_error_prob
        self.measure_channel_prob = measure_channel_prob
        self.rng = np.random.default_rng()

    def prepare_qubit(self) -> KrausChannel:
        """Return the channel to apply after clean single-qubit preparation. Here just identity."""
        return depolarising_channel(self.prepare_error_prob)

    def entangle(self) -> KrausChannel:
        """Return noise model to qubits that happens after the CZ gate."""
        # return two_qubit_depolarising_tensor_channel(self.entanglement_error_prob)
        return two_qubit_depolarising_channel(self.entanglement_error_prob)

    def measure(self) -> KrausChannel:
        """Apply noise to qubit to be measured."""
        return depolarising_channel(self.measure_channel_prob)

    def confuse_result(self, result: bool) -> bool:
        """Assign wrong measurement result cmd = "M"."""
        if self.rng.uniform() < self.measure_error_prob:
            return not result
        else:
            return result

    def byproduct_x(self) -> KrausChannel:
        """Apply noise to qubits after X gate correction."""
        return depolarising_channel(self.x_error_prob)

    def byproduct_z(self) -> KrausChannel:
        """Apply noise to qubits after Z gate correction."""
        return depolarising_channel(self.z_error_prob)

    def clifford(self) -> KrausChannel:
        """Apply noise to qubits that happens in the Clifford gate process."""
        # TODO list separate different Cliffords to allow customization
        return KrausChannel([KrausData(1.0, np.eye(2))])

    def tick_clock(self) -> None:
        """Notion of time in real devices - this is where we apply effect of T1 and T2.

        We assume commands that lie between 'T' commands run simultaneously on the device.
        """


class TestNoisyDensityMatrixBackend:
    """Test for Noisy DensityMatrixBackend simultation."""

    @staticmethod
    def rz_exact_res(alpha: float) -> npt.NDArray:
        return 0.5 * np.array([[1, np.exp(-1j * alpha)], [np.exp(1j * alpha), 1]])

    @staticmethod
    def hpat() -> Pattern:
        circ = Circuit(1)
        circ.h(0)
        return circ.transpile().pattern

    @staticmethod
    def rzpat(alpha: float) -> Pattern:
        circ = Circuit(1)
        circ.rz(0, alpha)
        return circ.transpile().pattern

    # test noiseless noisy vs noiseless
    @pytest.mark.filterwarnings("ignore:Simulating using densitymatrix backend with no noise.")
    def test_noiseless_noisy_hadamard(self, fx_rng: Generator) -> None:
        hadamardpattern = self.hpat()
        noiselessres = hadamardpattern.simulate_pattern(backend="densitymatrix", rng=fx_rng)
        # noiseless noise model
        noisynoiselessres = hadamardpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=NoiselessNoiseModel(),
            rng=fx_rng,
        )
        assert np.allclose(noiselessres.rho, np.array([[1.0, 0.0], [0.0, 0.0]]))
        # result should be |0>
        assert np.allclose(noisynoiselessres.rho, np.array([[1.0, 0.0], [0.0, 0.0]]))

    # test measurement confuse outcome
    def test_noisy_measure_confuse_hadamard(self, fx_rng: Generator) -> None:
        hadamardpattern = self.hpat()
        res = hadamardpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=NoiseModelTester(measure_error_prob=1.0),
            rng=fx_rng,
        )
        # result should be |1>
        assert np.allclose(res.rho, np.array([[0.0, 0.0], [0.0, 1.0]]))

        # arbitrary probability
        measure_error_pr = fx_rng.random()
        print(f"measure_error_pr = {measure_error_pr}")
        res = hadamardpattern.simulate_pattern(
            backend="densitymatrix", noise_model=NoiseModelTester(measure_error_prob=measure_error_pr), rng=fx_rng
        )
        # result should be |1>
        assert np.allclose(res.rho, np.array([[1.0, 0.0], [0.0, 0.0]])) or np.allclose(
            res.rho,
            np.array([[0.0, 0.0], [0.0, 1.0]]),
        )

    def test_noisy_measure_channel_hadamard(self, fx_rng: Generator) -> None:
        hadamardpattern = self.hpat()
        measure_channel_pr = fx_rng.random()
        print(f"measure_channel_pr = {measure_channel_pr}")
        # measurement error only
        res = hadamardpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=NoiseModelTester(measure_channel_prob=measure_channel_pr),
            rng=fx_rng,
        )
        # just TP the depolarizing channel
        assert np.allclose(
            res.rho,
            np.array([[1 - 2 * measure_channel_pr / 3.0, 0.0], [0.0, 2 * measure_channel_pr / 3.0]]),
        )

    # test Pauli X error
    def test_noisy_x_hadamard(self, fx_rng: Generator) -> None:
        hadamardpattern = self.hpat()
        # x error only
        x_error_pr = fx_rng.random()
        print(f"x_error_pr = {x_error_pr}")
        res = hadamardpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=NoiseModelTester(x_error_prob=x_error_pr),
            rng=fx_rng,
        )
        # analytical result since deterministic pattern output is |0>.
        # if no X applied, no noise. If X applied X noise on |0><0|

        assert np.allclose(res.rho, np.array([[1.0, 0.0], [0.0, 0.0]])) or np.allclose(
            res.rho,
            np.array([[1 - 2 * x_error_pr / 3.0, 0.0], [0.0, 2 * x_error_pr / 3.0]]),
        )

    # test entanglement error
    def test_noisy_entanglement_hadamard(self, fx_rng: Generator) -> None:
        hadamardpattern = self.hpat()
        entanglement_error_pr = fx_rng.uniform()
        res = hadamardpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=NoiseModelTester(entanglement_error_prob=entanglement_error_pr),
            rng=fx_rng,
        )
        # analytical result for tensor depolarizing channel
        # assert np.allclose(
        #     res.rho,
        #     np.array(
        #         [
        #             [1 - 4 * entanglement_error_pr / 3.0 + 8 * entanglement_error_pr**2 / 9, 0.0],
        #             [0.0, 4 * entanglement_error_pr / 3.0 - 8 * entanglement_error_pr**2 / 9],
        #         ]
        #     ),
        # )

        # analytical result for true 2-qubit depolarizing channel
        assert np.allclose(
            res.rho,
            np.array(
                [
                    [1 - 8 * entanglement_error_pr / 15.0, 0.0],
                    [0.0, 8 * entanglement_error_pr / 15.0],
                ],
            ),
        )

    # test preparation error
    def test_noisy_preparation_hadamard(self, fx_rng: Generator) -> None:
        hadamardpattern = self.hpat()
        prepare_error_pr = fx_rng.random()
        print(f"prepare_error_pr = {prepare_error_pr}")
        res = hadamardpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=NoiseModelTester(prepare_error_prob=prepare_error_pr),
            rng=fx_rng,
        )
        # analytical result
        assert np.allclose(
            res.rho,
            np.array([[1 - 2 * prepare_error_pr / 3.0, 0.0], [0.0, 2 * prepare_error_pr / 3.0]]),
        )

    # Test rz gate

    # test noiseless noisy vs noiseless
    @pytest.mark.filterwarnings("ignore:Simulating using densitymatrix backend with no noise.")
    def test_noiseless_noisy_rz(self, fx_rng: Generator) -> None:
        alpha = fx_rng.random()
        rzpattern = self.rzpat(alpha)
        noiselessres = rzpattern.simulate_pattern(backend="densitymatrix")
        # noiseless noise model or NoiseModelTester() since all probas are 0
        noisynoiselessres = rzpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=NoiseModelTester(),
            rng=fx_rng,
        )  # NoiselessNoiseModel()
        assert np.allclose(
            noiselessres.rho,
            0.5 * np.array([[1.0, np.exp(-1j * alpha)], [np.exp(1j * alpha), 1.0]]),
        )
        # result should be |0>
        assert np.allclose(
            noisynoiselessres.rho,
            0.5 * np.array([[1.0, np.exp(-1j * alpha)], [np.exp(1j * alpha), 1.0]]),
        )

    # test preparation error
    def test_noisy_preparation_rz(self, fx_rng: Generator) -> None:
        alpha = fx_rng.random()
        rzpattern = self.rzpat(alpha)
        prepare_error_pr = fx_rng.random()
        print(f"prepare_error_pr = {prepare_error_pr}")
        res = rzpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=NoiseModelTester(prepare_error_prob=prepare_error_pr),
            rng=fx_rng,
        )
        # analytical result
        assert np.allclose(
            res.rho,
            0.5
            * np.array(
                [
                    [
                        1.0,
                        (3 - 4 * prepare_error_pr) ** 2
                        * (3 * np.cos(alpha) + 1j * (-3 + 4 * prepare_error_pr) * np.sin(alpha))
                        / 27,
                    ],
                    [
                        (3 - 4 * prepare_error_pr) ** 2
                        * (3 * np.cos(alpha) - 1j * (-3 + 4 * prepare_error_pr) * np.sin(alpha))
                        / 27,
                        1.0,
                    ],
                ],
            ),
        )

    # test entanglement error
    def test_noisy_entanglement_rz(self, fx_rng: Generator) -> None:
        alpha = fx_rng.random()
        rzpattern = self.rzpat(alpha)
        entanglement_error_pr = fx_rng.uniform()
        res = rzpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=NoiseModelTester(entanglement_error_prob=entanglement_error_pr),
            rng=fx_rng,
        )
        # analytical result for tensor depolarizing channel
        # assert np.allclose(
        #     res.rho,
        #     0.5
        #     * np.array(
        #         [
        #             [
        #                 1.0,
        #                 (-3 + 4 * entanglement_error_pr) ** 3
        #                 * (-3 * np.cos(alpha) + 1j * (3 - 4 * entanglement_error_pr) * np.sin(alpha))
        #                 / 81,
        #             ],
        #             [
        #                 (-3 + 4 * entanglement_error_pr) ** 3
        #                 * (-3 * np.cos(alpha) - 1j * (3 - 4 * entanglement_error_pr) * np.sin(alpha))
        #                 / 81,
        #                 1.0,
        #             ],
        #         ]
        #     ),
        # )

        # analytical result for true 2-qubit depolarizing channel
        assert np.allclose(
            res.rho,
            0.5
            * np.array(
                [
                    [
                        1.0,
                        np.exp(-1j * alpha) * (15 - 16 * entanglement_error_pr) ** 2 / 225,
                    ],
                    [
                        np.exp(1j * alpha) * (15 - 16 * entanglement_error_pr) ** 2 / 225,
                        1.0,
                    ],
                ],
            ),
        )

    def test_noisy_measure_channel_rz(self, fx_rng: Generator) -> None:
        alpha = fx_rng.random()
        rzpattern = self.rzpat(alpha)
        measure_channel_pr = fx_rng.random()
        print(f"measure_channel_pr = {measure_channel_pr}")
        # measurement error only
        res = rzpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=NoiseModelTester(measure_channel_prob=measure_channel_pr),
            rng=fx_rng,
        )

        assert np.allclose(
            res.rho,
            0.5
            * np.array(
                [
                    [
                        1.0,
                        (-3 + 4 * measure_channel_pr)
                        * (-3 * np.cos(alpha) + 1j * (3 - 4 * measure_channel_pr) * np.sin(alpha))
                        / 9,
                    ],
                    [
                        (-3 + 4 * measure_channel_pr)
                        * (-3 * np.cos(alpha) - 1j * (3 - 4 * measure_channel_pr) * np.sin(alpha))
                        / 9,
                        1.0,
                    ],
                ],
            ),
        )

    def test_noisy_x_rz(self, fx_rng: Generator) -> None:
        alpha = fx_rng.random()
        rzpattern = self.rzpat(alpha)
        # x error only
        x_error_pr = fx_rng.random()
        print(f"x_error_pr = {x_error_pr}")
        res = rzpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=NoiseModelTester(x_error_prob=x_error_pr),
            rng=fx_rng,
        )

        # only two cases: if no X correction, Z or no Z correction but exact result.
        # If X correction the noise result is the same with or without the PERFECT Z correction.
        assert np.allclose(
            res.rho,
            0.5 * np.array([[1.0, np.exp(-1j * alpha)], [np.exp(1j * alpha), 1.0]]),
        ) or np.allclose(
            res.rho,
            0.5
            * np.array(
                [
                    [1.0, np.exp(-1j * alpha) * (3 - 4 * x_error_pr) / 3],
                    [np.exp(1j * alpha) * (3 - 4 * x_error_pr) / 3, 1.0],
                ],
            ),
        )

    def test_noisy_z_rz(self, fx_rng: Generator) -> None:
        alpha = fx_rng.random()
        rzpattern = self.rzpat(alpha)
        # z error only
        z_error_pr = fx_rng.random()
        print(f"z_error_pr = {z_error_pr}")
        res = rzpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=NoiseModelTester(z_error_prob=z_error_pr),
            rng=fx_rng,
        )

        # only two cases: if no Z correction, X or no X correction but exact result.
        # If Z correction the noise result is the same with or without the PERFECT X correction.
        assert np.allclose(
            res.rho,
            0.5 * np.array([[1.0, np.exp(-1j * alpha)], [np.exp(1j * alpha), 1.0]]),
        ) or np.allclose(
            res.rho,
            0.5
            * np.array(
                [
                    [1.0, np.exp(-1j * alpha) * (3 - 4 * z_error_pr) / 3],
                    [np.exp(1j * alpha) * (3 - 4 * z_error_pr) / 3, 1.0],
                ],
            ),
        )

    def test_noisy_xz_rz(self, fx_rng: Generator) -> None:
        alpha = fx_rng.random()
        rzpattern = self.rzpat(alpha)
        # x and z errors
        x_error_pr = fx_rng.random()
        print(f"x_error_pr = {x_error_pr}")
        z_error_pr = fx_rng.random()
        print(f"z_error_pr = {z_error_pr}")
        res = rzpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=NoiseModelTester(x_error_prob=x_error_pr, z_error_prob=z_error_pr),
            rng=fx_rng,
        )

        # 4 cases : no corr, noisy X, noisy Z, noisy XZ.
        assert (
            np.allclose(res.rho, 0.5 * np.array([[1.0, np.exp(-1j * alpha)], [np.exp(1j * alpha), 1.0]]))
            or np.allclose(
                res.rho,
                0.5
                * np.array(
                    [
                        [1.0, np.exp(-1j * alpha) * (3 - 4 * x_error_pr) * (3 - 4 * z_error_pr) / 9],
                        [np.exp(1j * alpha) * (3 - 4 * x_error_pr) * (3 - 4 * z_error_pr) / 9, 1.0],
                    ],
                ),
            )
            or np.allclose(
                res.rho,
                0.5
                * np.array(
                    [
                        [1.0, np.exp(-1j * alpha) * (3 - 4 * z_error_pr) / 3],
                        [np.exp(1j * alpha) * (3 - 4 * z_error_pr) / 3, 1.0],
                    ],
                ),
            )
            or np.allclose(
                res.rho,
                0.5
                * np.array(
                    [
                        [1.0, np.exp(-1j * alpha) * (3 - 4 * x_error_pr) / 3],
                        [np.exp(1j * alpha) * (3 - 4 * x_error_pr) / 3, 1.0],
                    ],
                ),
            )
        )

    # test measurement confuse outcome
    def test_noisy_measure_confuse_rz(self, fx_rng: Generator) -> None:
        alpha = fx_rng.random()
        rzpattern = self.rzpat(alpha)
        # probability 1 to shift both outcome
        res = rzpattern.simulate_pattern(
            backend="densitymatrix", noise_model=NoiseModelTester(measure_error_prob=1.0), rng=fx_rng
        )
        # result X, XZ or Z

        exact = self.rz_exact_res(alpha)

        assert (
            np.allclose(res.rho, Ops.X @ exact @ Ops.X)
            or np.allclose(res.rho, Ops.Z @ exact @ Ops.Z)
            or np.allclose(res.rho, Ops.Z @ Ops.X @ exact @ Ops.X @ Ops.Z)
        )

        # arbitrary probability
        measure_error_pr = fx_rng.random()
        print(f"measure_error_pr = {measure_error_pr}")
        res = rzpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=NoiseModelTester(measure_error_prob=measure_error_pr),
            rng=fx_rng,
        )
        # just add the case without readout errors
        assert (
            np.allclose(res.rho, exact)
            or np.allclose(res.rho, Ops.X @ exact @ Ops.X)
            or np.allclose(res.rho, Ops.Z @ exact @ Ops.Z)
            or np.allclose(res.rho, Ops.Z @ Ops.X @ exact @ Ops.X @ Ops.Z)
        )
