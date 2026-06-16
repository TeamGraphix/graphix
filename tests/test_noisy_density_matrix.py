from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest

from graphix.branch_selector import ConstBranchSelector, FixedBranchSelector
from graphix.channels import amplitude_damping_channel, two_qubit_amplitude_damping_channel
from graphix.command import CommandKind
from graphix.fundamentals import angle_to_rad
from graphix.measurements import Measurement
from graphix.noise_models import AmplitudeDampingNoiseModel, DepolarisingNoiseModel
from graphix.noise_models.noise_model import NoiselessNoiseModel
from graphix.ops import Ops
from graphix.sim.base_backend import _outcome_to_operator_matrix
from graphix.sim.density_matrix import DensityMatrix
from graphix.states import BasicStates
from graphix.transpiler import Circuit

if TYPE_CHECKING:
    from numpy.random import Generator

    from graphix.fundamentals import Angle
    from graphix.measurements import Outcome
    from graphix.pattern import Pattern


def rz_exact_res(alpha: Angle) -> npt.NDArray[np.float64]:
    rad = angle_to_rad(alpha)
    return 0.5 * np.array([[1, np.exp(-1j * rad)], [np.exp(1j * rad), 1]])


def single_qubit_amplitude_damping_exact(
    rho: npt.NDArray[np.complex128 | np.float64], gamma: float
) -> npt.NDArray[np.complex128]:
    """Apply the single-qubit amplitude damping channel to a 2x2 density matrix."""
    return np.array(
        [
            [rho[0, 0] + gamma * rho[1, 1], np.sqrt(1 - gamma) * rho[0, 1]],
            [np.sqrt(1 - gamma) * rho[1, 0], (1 - gamma) * rho[1, 1]],
        ],
        dtype=np.complex128,
    )


def rz_measurement_results(rzpattern: Pattern, z_outcome: Outcome, x_outcome: Outcome) -> dict[int, Outcome]:
    m_nodes = (cmd.node for cmd in rzpattern if cmd.kind == CommandKind.M)
    return {next(m_nodes): z_outcome, next(m_nodes): x_outcome}


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

    @pytest.mark.parametrize("outcome", [0, 1])
    def test_noisy_measure_confuse_hadamard_arbitrary(self, fx_rng: Generator, outcome: Outcome) -> None:
        # arbitrary probability with fixed branch
        hadamardpattern = hpat()
        measure_error_pr = fx_rng.random()
        print(f"measure_error_pr = {measure_error_pr}, outcome = {outcome}")
        res = hadamardpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=DepolarisingNoiseModel(measure_error_prob=measure_error_pr),
            branch_selector=ConstBranchSelector(outcome),
            rng=fx_rng,
        )
        assert isinstance(res, DensityMatrix)
        # With measure_error_prob, the outcome might be flipped, resulting in different X corrections
        # However, we cannot predict the exact result without knowing if the error occurred
        # So we check both possibilities
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
    @pytest.mark.parametrize("outcome", [0, 1])
    def test_noisy_x_hadamard(self, fx_rng: Generator, outcome: Outcome) -> None:
        hadamardpattern = hpat()
        # x error only
        x_error_pr = fx_rng.random()
        print(f"x_error_pr = {x_error_pr}, outcome = {outcome}")
        res = hadamardpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=DepolarisingNoiseModel(x_error_prob=x_error_pr),
            branch_selector=ConstBranchSelector(outcome),
            rng=fx_rng,
        )
        # Pattern has X(1, {0}), so X error noise only applied when outcome=1
        assert isinstance(res, DensityMatrix)
        if outcome == 0:
            # No X correction → no X error noise
            assert np.allclose(res.rho, np.array([[1.0, 0.0], [0.0, 0.0]]))
        else:
            # X correction applied → X error noise applied
            assert np.allclose(
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
        noiselessres = rzpattern.simulate_pattern(backend="densitymatrix", rng=fx_rng)
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

    @pytest.mark.parametrize("z_outcome", [0, 1])
    @pytest.mark.parametrize("x_outcome", [0, 1])
    def test_noisy_x_rz(self, fx_rng: Generator, z_outcome: Outcome, x_outcome: Outcome) -> None:
        alpha = fx_rng.random()
        rzpattern = rzpat(alpha)
        # x error only
        x_error_pr = fx_rng.random()
        print(f"x_error_pr = {x_error_pr}, outcome_z = {z_outcome}, outcome_x = {x_outcome}")

        # M(0) determines Z, M(1) determines X
        m_nodes = (cmd.node for cmd in rzpattern if cmd.kind == CommandKind.M)
        results: dict[int, Outcome] = {next(m_nodes): z_outcome, next(m_nodes): x_outcome}

        res = rzpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=DepolarisingNoiseModel(x_error_prob=x_error_pr),
            branch_selector=FixedBranchSelector(results),
            rng=fx_rng,
        )

        # Pattern has X(2, {1}), so X error noise only applied when x_outcome=1
        assert isinstance(res, DensityMatrix)
        rad = angle_to_rad(alpha)
        if x_outcome == 0:
            # No X correction → no X error noise
            assert np.allclose(res.rho, rz_exact_res(alpha))
        else:
            # X correction applied → X error noise applied
            assert np.allclose(
                res.rho,
                0.5
                * np.array(
                    [
                        [1.0, np.exp(-1j * rad) * (3 - 4 * x_error_pr) / 3],
                        [np.exp(1j * rad) * (3 - 4 * x_error_pr) / 3, 1.0],
                    ],
                ),
            )

    @pytest.mark.parametrize("outcome_z", [0, 1])
    @pytest.mark.parametrize("outcome_x", [0, 1])
    def test_noisy_z_rz(self, fx_rng: Generator, outcome_z: Outcome, outcome_x: Outcome) -> None:
        alpha = fx_rng.random()
        rzpattern = rzpat(alpha)
        # z error only
        z_error_pr = fx_rng.random()
        print(f"z_error_pr = {z_error_pr}, outcome_z = {outcome_z}, outcome_x = {outcome_x}")

        # M(0) determines Z, M(1) determines X
        results: dict[int, Outcome] = {0: outcome_z, 1: outcome_x}

        res = rzpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=DepolarisingNoiseModel(z_error_prob=z_error_pr),
            branch_selector=FixedBranchSelector(results),
            rng=fx_rng,
        )

        # Pattern has Z(2, {0}), so Z error noise only applied when outcome_z=1
        assert isinstance(res, DensityMatrix)
        rad = angle_to_rad(alpha)
        if outcome_z == 0:
            # No Z correction → no Z error noise
            assert np.allclose(res.rho, rz_exact_res(alpha))
        else:
            # Z correction applied → Z error noise applied
            assert np.allclose(
                res.rho,
                0.5
                * np.array(
                    [
                        [1.0, np.exp(-1j * rad) * (3 - 4 * z_error_pr) / 3],
                        [np.exp(1j * rad) * (3 - 4 * z_error_pr) / 3, 1.0],
                    ],
                ),
            )

    @pytest.mark.parametrize("z_outcome", [0, 1])
    @pytest.mark.parametrize("x_outcome", [0, 1])
    def test_noisy_xz_rz(self, fx_rng: Generator, z_outcome: Outcome, x_outcome: Outcome) -> None:
        alpha = fx_rng.random()
        rzpattern = rzpat(alpha)
        # x and z errors
        x_error_pr = fx_rng.random()
        print(f"x_error_pr = {x_error_pr}")
        z_error_pr = fx_rng.random()
        print(f"z_error_pr = {z_error_pr}")
        print(f"z_outcome = {z_outcome}, x_outcome = {x_outcome}")

        # M(0) determines Z correction, M(1) determines X correction
        results: dict[int, Outcome] = {0: z_outcome, 1: x_outcome}

        res = rzpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=DepolarisingNoiseModel(x_error_prob=x_error_pr, z_error_prob=z_error_pr),
            branch_selector=FixedBranchSelector(results),
            rng=fx_rng,
        )

        # Pattern has X(2, {1}) and Z(2, {0}), noise applied conditionally
        assert isinstance(res, DensityMatrix)
        rad = angle_to_rad(alpha)
        if z_outcome == 0 and x_outcome == 0:
            # No corrections → no noise
            assert np.allclose(res.rho, rz_exact_res(alpha))
        elif z_outcome == 0 and x_outcome == 1:
            # Only X correction → only X noise
            assert np.allclose(
                res.rho,
                0.5
                * np.array(
                    [
                        [1.0, np.exp(-1j * rad) * (3 - 4 * x_error_pr) / 3],
                        [np.exp(1j * rad) * (3 - 4 * x_error_pr) / 3, 1.0],
                    ],
                ),
            )
        elif z_outcome == 1 and x_outcome == 0:
            # Only Z correction → only Z noise
            assert np.allclose(
                res.rho,
                0.5
                * np.array(
                    [
                        [1.0, np.exp(-1j * rad) * (3 - 4 * z_error_pr) / 3],
                        [np.exp(1j * rad) * (3 - 4 * z_error_pr) / 3, 1.0],
                    ],
                ),
            )
        else:  # z_outcome == 1 and x_outcome == 1
            # Both corrections → both noises
            assert np.allclose(
                res.rho,
                0.5
                * np.array(
                    [
                        [1.0, np.exp(-1j * rad) * (3 - 4 * x_error_pr) * (3 - 4 * z_error_pr) / 9],
                        [np.exp(1j * rad) * (3 - 4 * x_error_pr) * (3 - 4 * z_error_pr) / 9, 1.0],
                    ],
                ),
            )

    # test measurement confuse outcome
    @pytest.mark.parametrize("z_outcome", [0, 1])
    @pytest.mark.parametrize("x_outcome", [0, 1])
    def test_noisy_measure_confuse_rz(self, fx_rng: Generator, z_outcome: Outcome, x_outcome: Outcome) -> None:
        alpha = fx_rng.random()
        rzpattern = rzpat(alpha)

        # M(0) determines Z, M(1) determines X
        results: dict[int, Outcome] = {0: z_outcome, 1: x_outcome}

        # Test with probability 1 to flip both outcomes
        res = rzpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=DepolarisingNoiseModel(measure_error_prob=1.0),
            branch_selector=FixedBranchSelector(results),
            rng=fx_rng,
        )

        exact = rz_exact_res(alpha)
        assert isinstance(res, DensityMatrix)
        # All outcomes lead to same result: both corrections applied due to flipping
        assert np.allclose(res.rho, Ops.Z @ Ops.X @ exact @ Ops.X @ Ops.Z)

    @pytest.mark.parametrize("z_outcome", [0, 1])
    @pytest.mark.parametrize("x_outcome", [0, 1])
    def test_noisy_measure_confuse_rz_arbitrary(
        self, fx_rng: Generator, z_outcome: Outcome, x_outcome: Outcome
    ) -> None:
        alpha = fx_rng.random()
        rzpattern = rzpat(alpha)

        # M(0) determines Z, M(1) determines X
        results: dict[int, Outcome] = {0: z_outcome, 1: x_outcome}

        # Test with arbitrary probability
        measure_error_pr = fx_rng.random()
        print(f"measure_error_pr = {measure_error_pr}, z_outcome = {z_outcome}, x_outcome = {x_outcome}")
        res = rzpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=DepolarisingNoiseModel(measure_error_prob=measure_error_pr),
            branch_selector=FixedBranchSelector(results),
            rng=fx_rng,
        )

        exact = rz_exact_res(alpha)
        assert isinstance(res, DensityMatrix)

        # With arbitrary measure_error_pr, outcomes may or may not be flipped
        # The physical result depends on whether the error occurs
        # We check all possible cases
        assert (
            np.allclose(res.rho, exact)
            or np.allclose(res.rho, Ops.X @ exact @ Ops.X)
            or np.allclose(res.rho, Ops.Z @ exact @ Ops.Z)
            or np.allclose(res.rho, Ops.Z @ Ops.X @ exact @ Ops.X @ Ops.Z)
        )


class TestAmplitudeDampingAnalytic:
    """Analytic per-step verification of amplitude damping noise."""

    @staticmethod
    def _kraus_operators(gamma: float) -> list[npt.NDArray[np.complex128]]:
        return [
            np.array([[1.0, 0.0], [0.0, np.sqrt(1 - gamma)]], dtype=np.complex128),
            np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=np.complex128),
        ]

    @staticmethod
    def apply_amplitude_damping_single_qubit(
        rho: npt.NDArray[np.complex128], gamma: float
    ) -> npt.NDArray[np.complex128]:
        return sum(
            (k @ rho @ k.conj().T for k in TestAmplitudeDampingAnalytic._kraus_operators(gamma)),
            np.zeros((2, 2), dtype=np.complex128),
        )

    @staticmethod
    def apply_amplitude_damping_on_qubit(
        rho: npt.NDArray[np.complex128], qubit: int, gamma: float
    ) -> npt.NDArray[np.complex128]:
        eye = np.eye(2, dtype=np.complex128)
        out = np.zeros(rho.shape, dtype=np.complex128)
        for kraus in TestAmplitudeDampingAnalytic._kraus_operators(gamma):
            full = np.kron(kraus, eye) if qubit == 0 else np.kron(eye, kraus)
            out += full @ rho @ full.conj().T
        return out

    @staticmethod
    def apply_amplitude_damping_two_qubit(rho: npt.NDArray[np.complex128], gamma: float) -> npt.NDArray[np.complex128]:
        out = np.zeros(rho.shape, dtype=np.complex128)
        for left in TestAmplitudeDampingAnalytic._kraus_operators(gamma):
            for right in TestAmplitudeDampingAnalytic._kraus_operators(gamma):
                full = np.kron(left, right)
                out += full @ rho @ full.conj().T
        return out

    @staticmethod
    def _apply_measurement_and_trace_out(
        dm: DensityMatrix, qubit: int, measurement: Measurement, outcome: Outcome
    ) -> None:
        bloch = measurement.to_bloch()
        vec = bloch.plane.polar(bloch.angle)
        op_mat = _outcome_to_operator_matrix(vec, outcome)
        dm.evolve_single(op_mat, qubit)
        dm.remove_qubit(qubit)

    def _expected_hadamard_pattern(self, step: str, gamma: float, outcome: Outcome) -> npt.NDArray[np.complex128]:
        eye = np.eye(2, dtype=np.complex128)
        rho: npt.NDArray[np.complex128] = np.asarray(
            DensityMatrix(data=[BasicStates.PLUS, BasicStates.PLUS]).rho,
            dtype=np.complex128,
        )

        if step == "prep":
            rho = self.apply_amplitude_damping_on_qubit(rho, 0, gamma)
            rho = self.apply_amplitude_damping_on_qubit(rho, 1, gamma)
        cz = np.asarray(Ops.CZ, dtype=np.complex128)
        rho = cz @ rho @ cz.conj().T
        if step == "entangle":
            rho = self.apply_amplitude_damping_two_qubit(rho, gamma)
        if step == "measure":
            rho = self.apply_amplitude_damping_on_qubit(rho, 0, gamma)
        x_projector = np.kron(
            np.asarray(
                DensityMatrix(data=BasicStates.PLUS if outcome == 0 else BasicStates.MINUS).rho,
                dtype=np.complex128,
            ),
            eye,
        )
        rho = x_projector @ rho @ x_projector.conj().T
        rho = rho / np.trace(rho)
        reduced: npt.NDArray[np.complex128] = np.einsum("ijik->jk", rho.reshape(2, 2, 2, 2))
        x_op = np.asarray(Ops.X, dtype=np.complex128)
        if outcome == 1:
            reduced = x_op @ reduced @ x_op.conj().T
        if step == "xcorr" and outcome == 1:
            reduced = self.apply_amplitude_damping_single_qubit(reduced, gamma)
        return reduced

    def _expected_rz_pattern(
        self,
        alpha: Angle,
        step: str,
        gamma: float,
        z_outcome: Outcome,
        x_outcome: Outcome,
    ) -> npt.NDArray[np.complex128]:
        dm = DensityMatrix(data=[BasicStates.PLUS, BasicStates.PLUS, BasicStates.PLUS])

        if step == "prep":
            for qubit in (0, 1, 2):
                dm.apply_channel(amplitude_damping_channel(gamma), [qubit])

        dm.entangle((0, 1))
        if step == "entangle":
            dm.apply_channel(two_qubit_amplitude_damping_channel(gamma), [0, 1])
        dm.entangle((1, 2))
        if step == "entangle":
            dm.apply_channel(two_qubit_amplitude_damping_channel(gamma), [1, 2])

        if step == "measure":
            dm.apply_channel(amplitude_damping_channel(gamma), [0])
        self._apply_measurement_and_trace_out(dm, 0, Measurement.XY(-alpha), z_outcome)

        if step == "measure":
            dm.apply_channel(amplitude_damping_channel(gamma), [0])
        self._apply_measurement_and_trace_out(dm, 0, Measurement.X, x_outcome)

        if x_outcome == 1:
            dm.evolve_single(np.asarray(Ops.X, dtype=np.complex128), 0)
        if step == "xcorr" and x_outcome == 1:
            dm.apply_channel(amplitude_damping_channel(gamma), [0])

        if z_outcome == 1:
            dm.evolve_single(np.asarray(Ops.Z, dtype=np.complex128), 0)
        if step == "zcorr" and z_outcome == 1:
            dm.apply_channel(amplitude_damping_channel(gamma), [0])

        return np.asarray(dm.rho, dtype=np.complex128)

    @pytest.mark.filterwarnings("ignore:Simulating using densitymatrix backend with no noise.")
    def test_noiseless_amplitude_damping_hadamard(self, fx_rng: Generator) -> None:
        hadamardpattern = hpat()
        noiselessres = hadamardpattern.simulate_pattern(backend="densitymatrix", rng=fx_rng)
        noisynoiselessres = hadamardpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=AmplitudeDampingNoiseModel(),
            rng=fx_rng,
        )
        assert isinstance(noiselessres, DensityMatrix)
        assert isinstance(noisynoiselessres, DensityMatrix)
        assert np.allclose(noiselessres.rho, np.array([[1.0, 0.0], [0.0, 0.0]]))
        assert np.allclose(noisynoiselessres.rho, np.array([[1.0, 0.0], [0.0, 0.0]]))

    @pytest.mark.filterwarnings("ignore:Simulating using densitymatrix backend with no noise.")
    def test_noiseless_amplitude_damping_rz(self, fx_rng: Generator) -> None:
        alpha = fx_rng.random()
        rzpattern = rzpat(alpha)
        noiselessres = rzpattern.simulate_pattern(backend="densitymatrix", rng=fx_rng)
        noisynoiselessres = rzpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=AmplitudeDampingNoiseModel(),
            rng=fx_rng,
        )
        assert isinstance(noiselessres, DensityMatrix)
        assert isinstance(noisynoiselessres, DensityMatrix)
        assert np.allclose(noiselessres.rho, rz_exact_res(alpha))
        assert np.allclose(noisynoiselessres.rho, rz_exact_res(alpha))

    def test_noisy_measure_confuse_hadamard(self, fx_rng: Generator) -> None:
        hadamardpattern = hpat()
        res = hadamardpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=AmplitudeDampingNoiseModel(measure_error_prob=1.0),
            rng=fx_rng,
        )
        assert isinstance(res, DensityMatrix)
        assert np.allclose(res.rho, np.array([[0.0, 0.0], [0.0, 1.0]]))

    @pytest.mark.parametrize("outcome", [0, 1])
    @pytest.mark.parametrize(
        ("step", "param"),
        [
            ("prep", "prepare_error_prob"),
            ("entangle", "entanglement_error_prob"),
            ("measure", "measure_channel_prob"),
            ("xcorr", "x_error_prob"),
        ],
    )
    def test_amplitude_damping_hadamard_step_matches_analytic(
        self, step: str, param: str, outcome: Outcome, fx_rng: Generator
    ) -> None:
        gamma = fx_rng.random()
        res = hpat().simulate_pattern(
            backend="densitymatrix",
            noise_model=AmplitudeDampingNoiseModel(**{param: gamma}),
            branch_selector=ConstBranchSelector(outcome),
            rng=fx_rng,
        )
        assert isinstance(res, DensityMatrix)
        assert np.allclose(res.rho, self._expected_hadamard_pattern(step, gamma, outcome))

        if step == "measure":
            sqrt_term = np.sqrt(1 - gamma)
            closed_form = np.array([[(1 + sqrt_term) / 2, 0.0], [0.0, (1 - sqrt_term) / 2]], dtype=np.complex128)
            assert np.allclose(res.rho, closed_form)

    @pytest.mark.parametrize("z_outcome", [0, 1])
    @pytest.mark.parametrize("x_outcome", [0, 1])
    @pytest.mark.parametrize(
        ("step", "param"),
        [
            ("prep", "prepare_error_prob"),
            ("entangle", "entanglement_error_prob"),
            ("measure", "measure_channel_prob"),
            ("xcorr", "x_error_prob"),
            ("zcorr", "z_error_prob"),
        ],
    )
    def test_amplitude_damping_rz_step_matches_analytic(
        self,
        step: str,
        param: str,
        z_outcome: Outcome,
        x_outcome: Outcome,
        fx_rng: Generator,
    ) -> None:
        alpha = fx_rng.random()
        gamma = fx_rng.random()
        rzpattern = rzpat(alpha)
        results = rz_measurement_results(rzpattern, z_outcome, x_outcome)

        res = rzpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=AmplitudeDampingNoiseModel(**{param: gamma}),
            branch_selector=FixedBranchSelector(results),
            rng=fx_rng,
        )
        assert isinstance(res, DensityMatrix)
        assert np.allclose(
            res.rho,
            self._expected_rz_pattern(alpha, step, gamma, z_outcome, x_outcome),
        )

    @pytest.mark.parametrize("z_outcome", [0, 1])
    @pytest.mark.parametrize("x_outcome", [0, 1])
    def test_amplitude_damping_x_rz(self, fx_rng: Generator, z_outcome: Outcome, x_outcome: Outcome) -> None:
        alpha = fx_rng.random()
        rzpattern = rzpat(alpha)
        gamma = fx_rng.random()
        results = rz_measurement_results(rzpattern, z_outcome, x_outcome)

        res = rzpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=AmplitudeDampingNoiseModel(x_error_prob=gamma),
            branch_selector=FixedBranchSelector(results),
            rng=fx_rng,
        )

        exact = rz_exact_res(alpha).astype(np.complex128)
        expected = single_qubit_amplitude_damping_exact(exact, gamma) if x_outcome == 1 else exact
        assert isinstance(res, DensityMatrix)
        assert np.allclose(res.rho, expected)

    @pytest.mark.parametrize("z_outcome", [0, 1])
    @pytest.mark.parametrize("x_outcome", [0, 1])
    def test_amplitude_damping_z_rz(self, fx_rng: Generator, z_outcome: Outcome, x_outcome: Outcome) -> None:
        alpha = fx_rng.random()
        rzpattern = rzpat(alpha)
        gamma = fx_rng.random()
        results = rz_measurement_results(rzpattern, z_outcome, x_outcome)

        res = rzpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=AmplitudeDampingNoiseModel(z_error_prob=gamma),
            branch_selector=FixedBranchSelector(results),
            rng=fx_rng,
        )

        exact = rz_exact_res(alpha).astype(np.complex128)
        expected = single_qubit_amplitude_damping_exact(exact, gamma) if z_outcome == 1 else exact
        assert isinstance(res, DensityMatrix)
        assert np.allclose(res.rho, expected)

    @pytest.mark.parametrize("z_outcome", [0, 1])
    @pytest.mark.parametrize("x_outcome", [0, 1])
    def test_amplitude_damping_measure_confuse_rz(
        self, fx_rng: Generator, z_outcome: Outcome, x_outcome: Outcome
    ) -> None:
        alpha = fx_rng.random()
        rzpattern = rzpat(alpha)
        results = rz_measurement_results(rzpattern, z_outcome, x_outcome)

        res = rzpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=AmplitudeDampingNoiseModel(measure_error_prob=1.0),
            branch_selector=FixedBranchSelector(results),
            rng=fx_rng,
        )

        exact = rz_exact_res(alpha)
        assert isinstance(res, DensityMatrix)
        assert np.allclose(res.rho, Ops.Z @ Ops.X @ exact @ Ops.X @ Ops.Z)
