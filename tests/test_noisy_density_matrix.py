from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest

from graphix.branch_selector import ConstBranchSelector, FixedBranchSelector
from graphix.command import CommandKind
from graphix.fundamentals import angle_to_rad
from graphix.noise_models import AmplitudeDampingNoiseModel, DepolarisingNoiseModel
from graphix.noise_models.noise_model import NoiselessNoiseModel
from graphix.ops import Ops
from graphix.sim.density_matrix import DensityMatrix
from graphix.transpiler import Circuit

if TYPE_CHECKING:
    from numpy.random import Generator

    from graphix.fundamentals import Angle
    from graphix.measurements import Outcome
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

    # --- Amplitude damping noise model tests ---

    def test_noiseless_amplitude_damping_hadamard(self, fx_rng: Generator) -> None:
        """AmplitudeDampingNoiseModel with all-zero defaults behaves like noiseless."""
        hadamardpattern = hpat()
        res = hadamardpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=AmplitudeDampingNoiseModel(),
            rng=fx_rng,
        )
        assert isinstance(res, DensityMatrix)
        assert np.allclose(res.rho, np.array([[1.0, 0.0], [0.0, 0.0]]))

    def test_amplitude_damping_noise_changes_hadamard(self, fx_rng: Generator) -> None:
        """Amplitude damping prepare noise changes the state from the noiseless result."""
        hadamardpattern = hpat()
        gamma = fx_rng.uniform(0.1, 0.5)
        res = hadamardpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=AmplitudeDampingNoiseModel(prepare_error_prob=gamma),
            rng=fx_rng,
        )
        assert isinstance(res, DensityMatrix)
        # State should be different from noiseless |0>
        assert not np.allclose(res.rho, np.array([[1.0, 0.0], [0.0, 0.0]]))
        # Density matrix should be valid: trace 1, Hermitian
        assert np.isclose(np.trace(res.rho), 1.0)
        assert np.allclose(res.rho, res.rho.conj().T)

    def test_amplitude_damping_confuse_measurement(self, fx_rng: Generator) -> None:
        """measure_error_prob=1 flips measurement outcome."""
        hadamardpattern = hpat()
        res = hadamardpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=AmplitudeDampingNoiseModel(measure_error_prob=1.0),
            rng=fx_rng,
        )
        assert isinstance(res, DensityMatrix)
        assert np.allclose(res.rho, np.array([[0.0, 0.0], [0.0, 1.0]]))

    # ----- Analytical comparison tests for amplitude damping -----

    @staticmethod
    def _ad_kraus(gamma: float) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
        """Return the two Kraus operators for the single-qubit amplitude damping channel."""
        k0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=np.complex128)
        k1 = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=np.complex128)
        return k0, k1

    @staticmethod
    def _apply_ad_2q(
        rho: npt.NDArray[np.complex128], gamma: float, qubit: int
    ) -> npt.NDArray[np.complex128]:
        """Apply amplitude damping with parameter gamma to one qubit of a 2-qubit density matrix."""
        k0, k1 = TestNoisyDensityMatrixBackend._ad_kraus(gamma)
        if qubit == 0:
            op0 = np.kron(k0, np.eye(2, dtype=np.complex128))
            op1 = np.kron(k1, np.eye(2, dtype=np.complex128))
        else:
            op0 = np.kron(np.eye(2, dtype=np.complex128), k0)
            op1 = np.kron(np.eye(2, dtype=np.complex128), k1)
        return op0 @ rho @ op0.conj().T + op1 @ rho @ op1.conj().T

    @staticmethod
    def _partial_trace_qubit0(
        rho: npt.NDArray[np.complex128],
    ) -> npt.NDArray[np.complex128]:
        """Trace out qubit 0 of a 2-qubit density matrix."""
        res = np.zeros((2, 2), dtype=np.complex128)
        for k in range(2):
            e = np.zeros(2, dtype=np.complex128)
            e[k] = 1.0
            proj = np.kron(e.reshape(-1, 1), np.eye(2, dtype=np.complex128))
            res += proj.conj().T @ rho @ proj
        return res

    @staticmethod
    def _amplitude_damping_measure_analytical(
        gamma: float, outcome: Outcome
    ) -> npt.NDArray[np.complex128]:
        """Compute expected hpat output for AmplitudeDampingNoiseModel(measure_channel_prob=gamma).

        Applies AD on node 0 after CZ, then measures in X basis, then
        applies X(1,{0}) correction when outcome is odd.
        """
        rho_plus = 0.5 * np.ones((2, 2), dtype=np.complex128)
        rho_plus[0, 1] = 0.5
        rho_plus[1, 0] = 0.5
        rho_initial = np.kron(rho_plus, rho_plus)

        cz = Ops.CZ
        rho_cz = cz @ rho_initial @ cz.conj().T

        rho_noisy = TestNoisyDensityMatrixBackend._apply_ad_2q(rho_cz, gamma, qubit=0)

        if outcome == 0:
            proj = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
        else:
            proj = np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=np.complex128)
        proj_full = np.kron(proj, np.eye(2, dtype=np.complex128))
        rho_meas = proj_full @ rho_noisy @ proj_full.conj().T
        prob = np.trace(rho_meas).real
        rho_out = TestNoisyDensityMatrixBackend._partial_trace_qubit0(rho_meas) / prob

        if outcome % 2 == 1:
            x_mat = Ops.X
            rho_out = x_mat @ rho_out @ x_mat.conj().T

        return rho_out

    @staticmethod
    def _amplitude_damping_prepare_analytical(
        gamma: float, outcome: Outcome
    ) -> npt.NDArray[np.complex128]:
        """Compute expected hpat output for AmplitudeDampingNoiseModel(prepare_error_prob=gamma).

        Applies AD to |+⟩ on each node independently, then CZ, measure, correct.
        """
        k0, k1 = TestNoisyDensityMatrixBackend._ad_kraus(gamma)
        rho_plus = 0.5 * np.ones((2, 2), dtype=np.complex128)
        rho_plus[0, 1] = 0.5
        rho_plus[1, 0] = 0.5

        rho_ad = k0 @ rho_plus @ k0.conj().T + k1 @ rho_plus @ k1.conj().T
        rho_prep = np.kron(rho_ad, rho_ad)

        cz = Ops.CZ
        rho_cz = cz @ rho_prep @ cz.conj().T

        if outcome == 0:
            proj = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
        else:
            proj = np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=np.complex128)
        proj_full = np.kron(proj, np.eye(2, dtype=np.complex128))
        rho_meas = proj_full @ rho_cz @ proj_full.conj().T
        prob = np.trace(rho_meas).real
        rho_out = TestNoisyDensityMatrixBackend._partial_trace_qubit0(rho_meas) / prob

        if outcome % 2 == 1:
            x_mat = Ops.X
            rho_out = x_mat @ rho_out @ x_mat.conj().T

        return rho_out

    @staticmethod
    def _amplitude_damping_entanglement_analytical(
        gamma: float, outcome: Outcome
    ) -> npt.NDArray[np.complex128]:
        """Compute expected hpat output for AmplitudeDampingNoiseModel(entanglement_error_prob=gamma).

        Applies independent AD to each qubit after CZ (tensor Kraus product),
        then measures and corrects.
        """
        rho_plus = 0.5 * np.ones((2, 2), dtype=np.complex128)
        rho_plus[0, 1] = 0.5
        rho_plus[1, 0] = 0.5
        rho_initial = np.kron(rho_plus, rho_plus)

        cz = Ops.CZ
        rho_cz = cz @ rho_initial @ cz.conj().T

        k0, k1 = TestNoisyDensityMatrixBackend._ad_kraus(gamma)
        kraus_2q = [
            np.kron(k0, k0),
            np.kron(k0, k1),
            np.kron(k1, k0),
            np.kron(k1, k1),
        ]
        rho_noisy = np.zeros_like(rho_cz, dtype=np.complex128)
        for op in kraus_2q:
            rho_noisy += op @ rho_cz @ op.conj().T

        if outcome == 0:
            proj = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
        else:
            proj = np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=np.complex128)
        proj_full = np.kron(proj, np.eye(2, dtype=np.complex128))
        rho_meas = proj_full @ rho_noisy @ proj_full.conj().T
        prob = np.trace(rho_meas).real
        rho_out = TestNoisyDensityMatrixBackend._partial_trace_qubit0(rho_meas) / prob

        if outcome % 2 == 1:
            x_mat = Ops.X
            rho_out = x_mat @ rho_out @ x_mat.conj().T

        return rho_out

    @staticmethod
    def _amplitude_damping_x_error_analytical(
        gamma: float, outcome: Outcome
    ) -> npt.NDArray[np.complex128]:
        """Compute expected hpat output for AmplitudeDampingNoiseModel(x_error_prob=gamma).

        AD is applied to the output node AFTER the X correction (per the
        noise model's cmds -> ApplyNoise ordering). For hpat the post-correction
        output is |0>, and AD is the identity on |0>.
        """
        rho_plus = 0.5 * np.ones((2, 2), dtype=np.complex128)
        rho_plus[0, 1] = 0.5
        rho_plus[1, 0] = 0.5
        rho_initial = np.kron(rho_plus, rho_plus)

        cz = Ops.CZ
        rho_cz = cz @ rho_initial @ cz.conj().T

        if outcome == 0:
            proj = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
        else:
            proj = np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=np.complex128)
        proj_full = np.kron(proj, np.eye(2, dtype=np.complex128))
        rho_meas = proj_full @ rho_cz @ proj_full.conj().T
        prob = np.trace(rho_meas).real
        rho_out = TestNoisyDensityMatrixBackend._partial_trace_qubit0(rho_meas) / prob

        if outcome % 2 == 1:
            x_mat = Ops.X
            rho_out = x_mat @ rho_out @ x_mat.conj().T

        k0, k1 = TestNoisyDensityMatrixBackend._ad_kraus(gamma)
        return k0 @ rho_out @ k0.conj().T + k1 @ rho_out @ k1.conj().T

    # --- Tests using analytical formulas ---

    @pytest.mark.parametrize("outcome", [0, 1])
    def test_amplitude_damping_measure_analytical(
        self, fx_rng: Generator, outcome: Outcome
    ) -> None:
        """Compare measure_channel simulation against analytical Kraus computation."""
        gamma = fx_rng.uniform(0.0, 0.9)
        hadamardpattern = hpat()
        expected = TestNoisyDensityMatrixBackend._amplitude_damping_measure_analytical(
            gamma, outcome
        )
        res = hadamardpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=AmplitudeDampingNoiseModel(measure_channel_prob=gamma),
            branch_selector=ConstBranchSelector(outcome),
            rng=fx_rng,
        )
        assert isinstance(res, DensityMatrix)
        assert np.allclose(res.rho, expected, atol=1e-10)

    @pytest.mark.parametrize("outcome", [0, 1])
    def test_amplitude_damping_prepare_analytical(
        self, fx_rng: Generator, outcome: Outcome
    ) -> None:
        """Compare prepare_error simulation against analytical Kraus computation."""
        gamma = fx_rng.uniform(0.0, 0.9)
        hadamardpattern = hpat()
        expected = TestNoisyDensityMatrixBackend._amplitude_damping_prepare_analytical(
            gamma, outcome
        )
        res = hadamardpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=AmplitudeDampingNoiseModel(prepare_error_prob=gamma),
            branch_selector=ConstBranchSelector(outcome),
            rng=fx_rng,
        )
        assert isinstance(res, DensityMatrix)
        assert np.allclose(res.rho, expected, atol=1e-10)

    @pytest.mark.parametrize("outcome", [0, 1])
    def test_amplitude_damping_entanglement_analytical(
        self, fx_rng: Generator, outcome: Outcome
    ) -> None:
        """Compare entanglement_error simulation against analytical Kraus computation."""
        gamma = fx_rng.uniform(0.0, 0.9)
        hadamardpattern = hpat()
        expected = TestNoisyDensityMatrixBackend._amplitude_damping_entanglement_analytical(
            gamma, outcome
        )
        res = hadamardpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=AmplitudeDampingNoiseModel(entanglement_error_prob=gamma),
            branch_selector=ConstBranchSelector(outcome),
            rng=fx_rng,
        )
        assert isinstance(res, DensityMatrix)
        assert np.allclose(res.rho, expected, atol=1e-10)

    @pytest.mark.parametrize("outcome", [0, 1])
    def test_amplitude_damping_x_error_analytical(
        self, fx_rng: Generator, outcome: Outcome
    ) -> None:
        """Compare x_error simulation against analytical Kraus computation.

        Note: for the hpat pattern, the post-correction state of the output
        node is always |0⟩, and AD does not affect |0⟩. So the output is
        always |0⟩ regardless of gamma.
        """
        gamma = fx_rng.uniform(0.0, 0.9)
        hadamardpattern = hpat()
        expected = TestNoisyDensityMatrixBackend._amplitude_damping_x_error_analytical(
            gamma, outcome
        )
        res = hadamardpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=AmplitudeDampingNoiseModel(x_error_prob=gamma),
            branch_selector=ConstBranchSelector(outcome),
            rng=fx_rng,
        )
        assert isinstance(res, DensityMatrix)
        assert np.allclose(res.rho, expected, atol=1e-10)
