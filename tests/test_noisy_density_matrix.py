from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest

from graphix.fundamentals import angle_to_rad
from graphix.noise_models import DepolarisingNoiseModel
from graphix.noise_models.noise_model import NoiselessNoiseModel
from graphix.ops import Ops
from graphix.sim.density_matrix import DensityMatrix
from graphix.transpiler import Circuit

if TYPE_CHECKING:
    from numpy.random import Generator

    from graphix.fundamentals import Angle
    from graphix.pattern import Pattern


def rz_exact_res(alpha: Angle) -> npt.NDArray[np.float64]:
    rad = angle_to_rad(alpha)
    return 0.5 * np.array([[1, np.exp(-1j * rad)], [np.exp(1j * rad), 1]])


def hpat() -> Pattern:
    circ = Circuit(1)
    circ.h(0)
    return circ.transpile().pattern


def rzpat(alpha: Angle) -> Pattern:
    circ = Circuit(1)
    circ.rz(0, alpha)
    return circ.transpile().pattern


class TestNoisyDensityMatrixBackend:
    """Test for Noisy DensityMatrixBackend simultation."""

    # test noiseless noisy vs noiseless
    @pytest.mark.filterwarnings("ignore:Simulating using densitymatrix backend with no noise.")
    def test_noiseless_noisy_hadamard(self, fx_rng: Generator) -> None:
        hadamardpattern = hpat()
        noiselessres = hadamardpattern.simulate_pattern(backend="densitymatrix", rng=fx_rng)
        # noiseless noise model
        noisynoiselessres = hadamardpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=NoiselessNoiseModel(),
            rng=fx_rng,
        )
        assert isinstance(noiselessres, DensityMatrix)
        assert np.allclose(noiselessres.rho, np.array([[1.0, 0.0], [0.0, 0.0]]))
        # result should be |0>
        assert isinstance(noisynoiselessres, DensityMatrix)
        assert np.allclose(noisynoiselessres.rho, np.array([[1.0, 0.0], [0.0, 0.0]]))

    # test measurement confuse outcome
    def test_noisy_measure_confuse_hadamard(self, fx_rng: Generator) -> None:
        hadamardpattern = hpat()
        res = hadamardpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=DepolarisingNoiseModel(measure_error_prob=1.0),
            rng=fx_rng,
        )
        # result should be |1>
        assert isinstance(res, DensityMatrix)
        assert np.allclose(res.rho, np.array([[0.0, 0.0], [0.0, 1.0]]))

        # arbitrary probability
        measure_error_pr = fx_rng.random()
        print(f"measure_error_pr = {measure_error_pr}")
        res = hadamardpattern.simulate_pattern(
            backend="densitymatrix", noise_model=DepolarisingNoiseModel(measure_error_prob=measure_error_pr), rng=fx_rng
        )
        # result should be |1>
        assert isinstance(res, DensityMatrix)
        assert np.allclose(res.rho, np.array([[1.0, 0.0], [0.0, 0.0]])) or np.allclose(
            res.rho,
            np.array([[0.0, 0.0], [0.0, 1.0]]),
        )

    def test_noisy_measure_channel_hadamard(self, fx_rng: Generator) -> None:
        hadamardpattern = hpat()
        measure_channel_pr = fx_rng.random()
        print(f"measure_channel_pr = {measure_channel_pr}")
        # measurement error only
        res = hadamardpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=DepolarisingNoiseModel(measure_channel_prob=measure_channel_pr),
            rng=fx_rng,
        )
        # just TP the depolarizing channel
        assert isinstance(res, DensityMatrix)
        assert np.allclose(
            res.rho,
            np.array([[1 - 2 * measure_channel_pr / 3.0, 0.0], [0.0, 2 * measure_channel_pr / 3.0]]),
        )

    # test Pauli X error
    def test_noisy_x_hadamard(self, fx_rng: Generator) -> None:
        hadamardpattern = hpat()
        # x error only
        x_error_pr = fx_rng.random()
        print(f"x_error_pr = {x_error_pr}")
        res = hadamardpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=DepolarisingNoiseModel(x_error_prob=x_error_pr),
            rng=fx_rng,
        )
        # analytical result since deterministic pattern output is |0>.
        # if no X applied, no noise. If X applied X noise on |0><0|

        assert isinstance(res, DensityMatrix)
        assert np.allclose(res.rho, np.array([[1.0, 0.0], [0.0, 0.0]])) or np.allclose(
            res.rho,
            np.array([[1 - 2 * x_error_pr / 3.0, 0.0], [0.0, 2 * x_error_pr / 3.0]]),
        )

    # test entanglement error
    def test_noisy_entanglement_hadamard(self, fx_rng: Generator) -> None:
        hadamardpattern = hpat()
        entanglement_error_pr = fx_rng.uniform()
        res = hadamardpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=DepolarisingNoiseModel(entanglement_error_prob=entanglement_error_pr),
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
        assert isinstance(res, DensityMatrix)
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
        hadamardpattern = hpat()
        prepare_error_pr = fx_rng.random()
        print(f"prepare_error_pr = {prepare_error_pr}")
        res = hadamardpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=DepolarisingNoiseModel(prepare_error_prob=prepare_error_pr),
            rng=fx_rng,
        )
        # analytical result
        assert isinstance(res, DensityMatrix)
        assert np.allclose(
            res.rho,
            np.array([[1 - 2 * prepare_error_pr / 3.0, 0.0], [0.0, 2 * prepare_error_pr / 3.0]]),
        )

    # Test rz gate

    # test noiseless noisy vs noiseless
    @pytest.mark.filterwarnings("ignore:Simulating using densitymatrix backend with no noise.")
    def test_noiseless_noisy_rz(self, fx_rng: Generator) -> None:
        alpha = fx_rng.random()
        rzpattern = rzpat(alpha)
        noiselessres = rzpattern.simulate_pattern(backend="densitymatrix")
        # noiseless noise model or DepolarisingNoiseModel() since all probas are 0
        noisynoiselessres = rzpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=DepolarisingNoiseModel(),
            rng=fx_rng,
        )  # NoiselessNoiseModel()
        assert isinstance(noiselessres, DensityMatrix)
        assert np.allclose(noiselessres.rho, rz_exact_res(alpha))
        # result should be |0>
        assert isinstance(noisynoiselessres, DensityMatrix)
        assert np.allclose(noisynoiselessres.rho, rz_exact_res(alpha))

    # test preparation error
    def test_noisy_preparation_rz(self, fx_rng: Generator) -> None:
        alpha = fx_rng.random()
        rzpattern = rzpat(alpha)
        prepare_error_pr = fx_rng.random()
        print(f"prepare_error_pr = {prepare_error_pr}")
        res = rzpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=DepolarisingNoiseModel(prepare_error_prob=prepare_error_pr),
            rng=fx_rng,
        )
        # analytical result
        assert isinstance(res, DensityMatrix)
        rad = angle_to_rad(alpha)
        assert np.allclose(
            res.rho,
            0.5
            * np.array(
                [
                    [
                        1.0,
                        (3 - 4 * prepare_error_pr) ** 2
                        * (3 * np.cos(rad) + 1j * (-3 + 4 * prepare_error_pr) * np.sin(rad))
                        / 27,
                    ],
                    [
                        (3 - 4 * prepare_error_pr) ** 2
                        * (3 * np.cos(rad) - 1j * (-3 + 4 * prepare_error_pr) * np.sin(rad))
                        / 27,
                        1.0,
                    ],
                ],
            ),
        )

    # test entanglement error
    def test_noisy_entanglement_rz(self, fx_rng: Generator) -> None:
        alpha = fx_rng.random()
        rzpattern = rzpat(alpha)
        entanglement_error_pr = fx_rng.uniform()
        res = rzpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=DepolarisingNoiseModel(entanglement_error_prob=entanglement_error_pr),
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
        assert isinstance(res, DensityMatrix)
        rad = angle_to_rad(alpha)
        assert np.allclose(
            res.rho,
            0.5
            * np.array(
                [
                    [
                        1.0,
                        np.exp(-1j * rad) * (15 - 16 * entanglement_error_pr) ** 2 / 225,
                    ],
                    [
                        np.exp(1j * rad) * (15 - 16 * entanglement_error_pr) ** 2 / 225,
                        1.0,
                    ],
                ],
            ),
        )

    def test_noisy_measure_channel_rz(self, fx_rng: Generator) -> None:
        alpha = fx_rng.random()
        rzpattern = rzpat(alpha)
        measure_channel_pr = fx_rng.random()
        print(f"measure_channel_pr = {measure_channel_pr}")
        # measurement error only
        res = rzpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=DepolarisingNoiseModel(measure_channel_prob=measure_channel_pr),
            rng=fx_rng,
        )

        assert isinstance(res, DensityMatrix)
        rad = angle_to_rad(alpha)
        assert np.allclose(
            res.rho,
            0.5
            * np.array(
                [
                    [
                        1.0,
                        (-3 + 4 * measure_channel_pr)
                        * (-3 * np.cos(rad) + 1j * (3 - 4 * measure_channel_pr) * np.sin(rad))
                        / 9,
                    ],
                    [
                        (-3 + 4 * measure_channel_pr)
                        * (-3 * np.cos(rad) - 1j * (3 - 4 * measure_channel_pr) * np.sin(rad))
                        / 9,
                        1.0,
                    ],
                ],
            ),
        )

    def test_noisy_x_rz(self, fx_rng: Generator) -> None:
        alpha = fx_rng.random()
        rzpattern = rzpat(alpha)
        # x error only
        x_error_pr = fx_rng.random()
        print(f"x_error_pr = {x_error_pr}")
        res = rzpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=DepolarisingNoiseModel(x_error_prob=x_error_pr),
            rng=fx_rng,
        )

        # only two cases: if no X correction, Z or no Z correction but exact result.
        # If X correction the noise result is the same with or without the PERFECT Z correction.
        assert isinstance(res, DensityMatrix)
        rad = angle_to_rad(alpha)
        assert np.allclose(
            res.rho,
            0.5 * np.array([[1.0, np.exp(-1j * rad)], [np.exp(1j * rad), 1.0]]),
        ) or np.allclose(
            res.rho,
            0.5
            * np.array(
                [
                    [1.0, np.exp(-1j * rad) * (3 - 4 * x_error_pr) / 3],
                    [np.exp(1j * rad) * (3 - 4 * x_error_pr) / 3, 1.0],
                ],
            ),
        )

    def test_noisy_z_rz(self, fx_rng: Generator) -> None:
        alpha = fx_rng.random()
        rzpattern = rzpat(alpha)
        # z error only
        z_error_pr = fx_rng.random()
        print(f"z_error_pr = {z_error_pr}")
        res = rzpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=DepolarisingNoiseModel(z_error_prob=z_error_pr),
            rng=fx_rng,
        )

        # only two cases: if no Z correction, X or no X correction but exact result.
        # If Z correction the noise result is the same with or without the PERFECT X correction.
        assert isinstance(res, DensityMatrix)
        rad = angle_to_rad(alpha)
        assert np.allclose(
            res.rho,
            0.5 * np.array([[1.0, np.exp(-1j * rad)], [np.exp(1j * rad), 1.0]]),
        ) or np.allclose(
            res.rho,
            0.5
            * np.array(
                [
                    [1.0, np.exp(-1j * rad) * (3 - 4 * z_error_pr) / 3],
                    [np.exp(1j * rad) * (3 - 4 * z_error_pr) / 3, 1.0],
                ],
            ),
        )

    def test_noisy_xz_rz(self, fx_rng: Generator) -> None:
        alpha = fx_rng.random()
        rzpattern = rzpat(alpha)
        # x and z errors
        x_error_pr = fx_rng.random()
        print(f"x_error_pr = {x_error_pr}")
        z_error_pr = fx_rng.random()
        print(f"z_error_pr = {z_error_pr}")
        res = rzpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=DepolarisingNoiseModel(x_error_prob=x_error_pr, z_error_prob=z_error_pr),
            rng=fx_rng,
        )

        # 4 cases : no corr, noisy X, noisy Z, noisy XZ.
        assert isinstance(res, DensityMatrix)
        rad = angle_to_rad(alpha)
        assert (
            np.allclose(res.rho, 0.5 * np.array([[1.0, np.exp(-1j * rad)], [np.exp(1j * rad), 1.0]]))
            or np.allclose(
                res.rho,
                0.5
                * np.array(
                    [
                        [1.0, np.exp(-1j * rad) * (3 - 4 * x_error_pr) * (3 - 4 * z_error_pr) / 9],
                        [np.exp(1j * rad) * (3 - 4 * x_error_pr) * (3 - 4 * z_error_pr) / 9, 1.0],
                    ],
                ),
            )
            or np.allclose(
                res.rho,
                0.5
                * np.array(
                    [
                        [1.0, np.exp(-1j * rad) * (3 - 4 * z_error_pr) / 3],
                        [np.exp(1j * rad) * (3 - 4 * z_error_pr) / 3, 1.0],
                    ],
                ),
            )
            or np.allclose(
                res.rho,
                0.5
                * np.array(
                    [
                        [1.0, np.exp(-1j * rad) * (3 - 4 * x_error_pr) / 3],
                        [np.exp(1j * rad) * (3 - 4 * x_error_pr) / 3, 1.0],
                    ],
                ),
            )
        )

    # test measurement confuse outcome
    def test_noisy_measure_confuse_rz(self, fx_rng: Generator) -> None:
        alpha = fx_rng.random()
        rzpattern = rzpat(alpha)
        # probability 1 to shift both outcome
        res = rzpattern.simulate_pattern(
            backend="densitymatrix", noise_model=DepolarisingNoiseModel(measure_error_prob=1.0), rng=fx_rng
        )
        # result X, XZ or Z

        exact = rz_exact_res(alpha)

        assert isinstance(res, DensityMatrix)
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
            noise_model=DepolarisingNoiseModel(measure_error_prob=measure_error_pr),
            rng=fx_rng,
        )
        # just add the case without readout errors
        assert isinstance(res, DensityMatrix)
        assert (
            np.allclose(res.rho, exact)
            or np.allclose(res.rho, Ops.X @ exact @ Ops.X)
            or np.allclose(res.rho, Ops.Z @ exact @ Ops.Z)
            or np.allclose(res.rho, Ops.Z @ Ops.X @ exact @ Ops.X @ Ops.Z)
        )
