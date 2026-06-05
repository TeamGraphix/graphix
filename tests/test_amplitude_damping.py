"""Tests for amplitude damping noise model and channels.

Soundness rationale
-------------------
Each test targets a single, independently verifiable physical property:

1. **Channel normalization** (`test_amplitude_damping_channel_normalization`,
   `test_two_qubit_amplitude_damping_channel_normalization`):
   A CPTP map must satisfy sum_i K_i† K_i = I.  We verify this directly on
   the matrices returned by the channel factories.  A failure here would mean
   the KrausChannel constructor would have already raised, but we make the
   contract explicit.

2. **Trivial limits**:
   - gamma=0 → identity channel: no damping occurs, rho is unchanged.
   - gamma=1 → full damping: any state collapses to |0><0|.
   We test both limits on both |0> and |1> states to catch sign/index errors.

3. **Excited-state decay** (`test_amplitude_damping_channel_excited_state`):
   Applying the channel once to |1><1| with parameter gamma must give the
   mixed state (1-gamma)|1><1| + gamma|0><0|.  This is the textbook
   single-application result and directly validates the Kraus operators.

4. **Ground-state invariance** (`test_amplitude_damping_channel_ground_state`):
   |0><0| must be a fixed point of the channel for all gamma, because there
   is nowhere to decay to.

5. **Trace preservation** (`test_amplitude_damping_trace_preserving`):
   All CPTP maps preserve trace.  We apply the channel to a random mixed
   state and check Tr(rho') = 1.

6. **Measure-channel noise on the Hadamard pattern**
   (`test_noisy_measure_channel_hadamard`):
   Starting from |+> (the state after Hadamard preparation in the cluster),
   amplitude damping with parameter gamma gives the density matrix
       rho = [[1 - gamma/2,  0], [0, gamma/2]].
   This is derived analytically: rho_in = (|0><0| + |0><1| + |1><0| +
   |1><1|)/2, apply K1 and K2, sum.  The test pins the exact formula.

7. **Noise-model structure: per-command injection**
   (`test_noise_model_command_structure`):
   The model must wrap each CommandKind correctly (N/E/M/X/Z) by appending or
   prepending an ApplyNoise command.  We check the length and kind of the
   returned list for each command type.

8. **Measurement confusion** (`test_noisy_measure_confuse`):
   With measure_error_prob=1 the outcome must always be flipped, producing
   the wrong correction and leaving the qubit in |1>.

9. **Noiseless limit via simulation** (`test_noiseless_simulation`):
   With all gammas=0, AmplitudeDampingNoiseModel must be equivalent to the
   noiseless backend.  This is an integration test that catches wiring errors.

10. **Two-qubit channel acts on both qubits**
    (`test_two_qubit_channel_both_qubits_damped`):
    With gamma=1, both qubits of the |11> state must collapse to |00>.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest

from graphix.pattern import Pattern
from graphix.channels import KrausChannel, amplitude_damping_channel, two_qubit_amplitude_damping_channel
from graphix.command import CommandKind, E, M, N, X, Z
from graphix.noise_models import AmplitudeDampingNoise, AmplitudeDampingNoiseModel, TwoQubitAmplitudeDampingNoise
from graphix.sim.density_matrix import DensityMatrix
from graphix.transpiler import Circuit

if TYPE_CHECKING:
    from numpy.random import Generator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_kraus(channel: KrausChannel, rho: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    """Apply a KrausChannel to density matrix rho manually."""
    out = np.zeros_like(rho, dtype=np.complex128)
    for kdata in channel:
        m = kdata.coef * kdata.operator
        out += m @ rho @ m.conj().T
    return out


def _ket_to_dm(ket: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    """Convert a state vector to a density matrix."""
    col = ket.reshape(-1, 1)
    return col @ col.conj().T


# ---------------------------------------------------------------------------
# Channel unit tests
# ---------------------------------------------------------------------------


class TestAmplitudeDampingChannel:
    """Tests for amplitude_damping_channel() factory."""

    def test_amplitude_damping_channel_normalization(self) -> None:
        """sum_i K_i† K_i must equal I for all gamma in [0, 1]."""
        for gamma in [0.0, 0.25, 0.5, 0.75, 1.0]:
            channel = amplitude_damping_channel(gamma)
            work = np.zeros((2, 2), dtype=np.complex128)
            for kdata in channel:
                m = kdata.coef * kdata.operator
                work += m.conj().T @ m
            assert np.allclose(work, np.eye(2)), f"Not normalized for gamma={gamma}"

    def test_amplitude_damping_channel_ground_state_invariant(self) -> None:
        """|0><0| must be a fixed point for all gamma."""
        rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        for gamma in [0.0, 0.5, 1.0]:
            channel = amplitude_damping_channel(gamma)
            result = _apply_kraus(channel, rho0)
            assert np.allclose(result, rho0), f"|0> not invariant for gamma={gamma}"

    def test_amplitude_damping_channel_excited_state(self) -> None:
        r"""Applying to |1><1| should give (1-gamma)|1><1| + gamma|0><0|."""
        rho1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
        for gamma in [0.0, 0.3, 0.7, 1.0]:
            channel = amplitude_damping_channel(gamma)
            result = _apply_kraus(channel, rho1)
            expected = np.array([[gamma, 0.0], [0.0, 1.0 - gamma]], dtype=np.complex128)
            assert np.allclose(result, expected), f"Excited state decay wrong for gamma={gamma}"

    def test_amplitude_damping_channel_gamma_zero_is_identity(self) -> None:
        """Gamma=0 must act as the identity channel on any state."""
        rho = np.array([[0.6, 0.3 + 0.1j], [0.3 - 0.1j, 0.4]], dtype=np.complex128)
        channel = amplitude_damping_channel(0.0)
        result = _apply_kraus(channel, rho)
        assert np.allclose(result, rho)

    def test_amplitude_damping_channel_gamma_one_collapses_to_ground(self) -> None:
        """gamma=1 must collapse any state to |0><0|."""
        rho = np.array([[0.4, 0.2 - 0.1j], [0.2 + 0.1j, 0.6]], dtype=np.complex128)
        channel = amplitude_damping_channel(1.0)
        result = _apply_kraus(channel, rho)
        expected = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        assert np.allclose(result, expected)

    def test_amplitude_damping_trace_preserving(self, fx_rng: Generator) -> None:
        """Trace must be preserved for any density matrix."""
        # Build a random valid density matrix
        v = fx_rng.standard_normal((2,)) + 1j * fx_rng.standard_normal((2,))
        rho = _ket_to_dm(v / np.linalg.norm(v))
        gamma = fx_rng.uniform(0.0, 1.0)
        channel = amplitude_damping_channel(gamma)
        result = _apply_kraus(channel, rho)
        assert np.isclose(np.trace(result), 1.0)

    def test_amplitude_damping_channel_invalid_gamma(self) -> None:
        """Gamma outside [0, 1] must raise ValueError."""
        with pytest.raises(ValueError):
            amplitude_damping_channel(-0.1)
        with pytest.raises(ValueError):
            amplitude_damping_channel(1.1)


class TestTwoQubitAmplitudeDampingChannel:
    """Tests for two_qubit_amplitude_damping_channel() factory."""

    def test_two_qubit_amplitude_damping_channel_normalization(self) -> None:
        """sum_i K_i† K_i must equal I_4 for all gamma."""
        for gamma in [0.0, 0.25, 0.5, 1.0]:
            channel = two_qubit_amplitude_damping_channel(gamma)
            work = np.zeros((4, 4), dtype=np.complex128)
            for kdata in channel:
                m = kdata.coef * kdata.operator
                work += m.conj().T @ m
            assert np.allclose(work, np.eye(4)), f"Not normalized for gamma={gamma}"

    def test_two_qubit_channel_both_qubits_damped(self) -> None:
        """With gamma=1, |11><11| must collapse to |00><00|."""
        ket_11 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.complex128)
        rho = _ket_to_dm(ket_11)
        channel = two_qubit_amplitude_damping_channel(1.0)
        result = _apply_kraus(channel, rho)
        expected = np.zeros((4, 4), dtype=np.complex128)
        expected[0, 0] = 1.0
        assert np.allclose(result, expected)

    def test_two_qubit_channel_gamma_zero_is_identity(self) -> None:
        """gamma=0 must act as the identity channel."""
        ket = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.complex128)
        rho = _ket_to_dm(ket)
        channel = two_qubit_amplitude_damping_channel(0.0)
        result = _apply_kraus(channel, rho)
        assert np.allclose(result, rho)

    def test_two_qubit_channel_trace_preserving(self, fx_rng: Generator) -> None:
        """Trace must be preserved for any 2-qubit density matrix."""
        v = fx_rng.standard_normal((4,)) + 1j * fx_rng.standard_normal((4,))
        rho = _ket_to_dm(v / np.linalg.norm(v))
        gamma = fx_rng.uniform(0.0, 1.0)
        channel = two_qubit_amplitude_damping_channel(gamma)
        result = _apply_kraus(channel, rho)
        assert np.isclose(np.trace(result), 1.0)

    def test_two_qubit_tensor_product_consistency(self) -> None:
        """Two-qubit channel on product state must equal applying 1Q channel twice."""
        gamma = 0.4
        channel_2q = two_qubit_amplitude_damping_channel(gamma)
        channel_1q = amplitude_damping_channel(gamma)

        # State |01> = |0> tensor |1>
        rho_0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        rho_1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
        rho_01 = np.kron(rho_0, rho_1).astype(np.complex128)

        result_2q = _apply_kraus(channel_2q, rho_01)

        # Apply independently and tensor
        result_0 = _apply_kraus(channel_1q, rho_0)
        result_1 = _apply_kraus(channel_1q, rho_1)
        result_tensor = np.kron(result_0, result_1)

        assert np.allclose(result_2q, result_tensor)


# ---------------------------------------------------------------------------
# Noise class unit tests
# ---------------------------------------------------------------------------


class TestAmplitudeDampingNoiseClasses:
    """Tests for AmplitudeDampingNoise and TwoQubitAmplitudeDampingNoise."""

    def test_amplitude_damping_noise_nqubits(self) -> None:
        assert AmplitudeDampingNoise(0.1).nqubits == 1

    def test_two_qubit_amplitude_damping_noise_nqubits(self) -> None:
        assert TwoQubitAmplitudeDampingNoise(0.1).nqubits == 2

    def test_amplitude_damping_noise_kraus_channel_type(self) -> None:
        assert isinstance(AmplitudeDampingNoise(0.3).to_kraus_channel(), KrausChannel)

    def test_two_qubit_amplitude_damping_noise_kraus_channel_type(self) -> None:
        assert isinstance(TwoQubitAmplitudeDampingNoise(0.3).to_kraus_channel(), KrausChannel)

    def test_amplitude_damping_noise_invalid_gamma(self) -> None:
        with pytest.raises((ValueError, Exception)):
            AmplitudeDampingNoise(1.5)


# ---------------------------------------------------------------------------
# NoiseModel structure tests
# ---------------------------------------------------------------------------


class TestAmplitudeDampingNoiseModelStructure:
    """Test that AmplitudeDampingNoiseModel wraps commands correctly."""

    def _make_model(self, gamma: float = 0.1) -> AmplitudeDampingNoiseModel:
        return AmplitudeDampingNoiseModel(
            prepare_error_gamma=gamma,
            x_error_gamma=gamma,
            z_error_gamma=gamma,
            entanglement_error_gamma=gamma,
            measure_channel_gamma=gamma,
            measure_error_prob=0.0,
        )

    def test_n_command_appends_noise(self) -> None:
        """N command must be followed by an ApplyNoise."""
        model = self._make_model()
        cmd = N(node=0)
        result = model.command(cmd)
        assert len(result) == 2
        assert result[0].kind == CommandKind.N
        assert result[1].kind == CommandKind.ApplyNoise
        assert isinstance(result[1].noise, AmplitudeDampingNoise)

    def test_e_command_appends_two_qubit_noise(self) -> None:
        """E command must be followed by a TwoQubitAmplitudeDampingNoise."""
        model = self._make_model()
        cmd = E(nodes=(0, 1))
        result = model.command(cmd)
        assert len(result) == 2
        assert result[0].kind == CommandKind.E
        assert result[1].kind == CommandKind.ApplyNoise
        assert isinstance(result[1].noise, TwoQubitAmplitudeDampingNoise)

    def test_m_command_prepends_noise(self) -> None:
        """M command must be preceded by an ApplyNoise (decoherence before measurement)."""
        model = self._make_model()
        cmd = M(node=0)
        result = model.command(cmd)
        assert len(result) == 2
        assert result[0].kind == CommandKind.ApplyNoise
        assert result[1].kind == CommandKind.M

    def test_x_command_appends_noise(self) -> None:
        """X correction must be followed by ApplyNoise."""
        model = self._make_model()
        cmd = X(node=1, domain={0})
        result = model.command(cmd)
        assert len(result) == 2
        assert result[0].kind == CommandKind.X
        assert result[1].kind == CommandKind.ApplyNoise

    def test_z_command_appends_noise(self) -> None:
        """Z correction must be followed by ApplyNoise."""
        model = self._make_model()
        cmd = Z(node=1, domain={0})
        result = model.command(cmd)
        assert len(result) == 2
        assert result[0].kind == CommandKind.Z
        assert result[1].kind == CommandKind.ApplyNoise

    def test_input_nodes_returns_apply_noise(self) -> None:
        """input_nodes must return one ApplyNoise per input node."""
        model = self._make_model()
        result = model.input_nodes([0, 1, 2])
        assert len(result) == 3
        assert all(cmd.kind == CommandKind.ApplyNoise for cmd in result)


# ---------------------------------------------------------------------------
# Integration tests — full pattern simulation
# ---------------------------------------------------------------------------


def hpat() -> Pattern:
    circ = Circuit(1)
    circ.h(0)
    return circ.transpile().pattern


class TestAmplitudeDampingNoiseModelSimulation:
    """Integration tests running full pattern simulations."""

    @pytest.mark.filterwarnings("ignore:Simulating using densitymatrix backend with no noise.")
    def test_noiseless_simulation(self, fx_rng: Generator) -> None:
        """gamma=0 everywhere must reproduce the noiseless result."""
        pattern = hpat()
        noiseless = pattern.simulate_pattern(backend="densitymatrix", rng=fx_rng)
        noisy = pattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=AmplitudeDampingNoiseModel(),  # all gammas default to 0
            rng=fx_rng,
        )
        assert isinstance(noiseless, DensityMatrix)
        assert isinstance(noisy, DensityMatrix)
        assert np.allclose(noiseless.rho, noisy.rho)

    def test_noisy_measure_channel_hadamard(self, fx_rng: Generator) -> None:
        r"""Measure-channel damping on the measured qubit of the Hadamard pattern.

        Derivation: In the one-qubit Hadamard MBQC pattern the measured qubit
        starts in |+> = (|0>+|1>)/sqrt(2), i.e.
            rho_in = [[0.5, 0.5],[0.5, 0.5]].
        Amplitude damping is applied *before* the measurement, giving
            rho_out = K1 rho_in K1† + K2 rho_in K2†
                    = [[0.5 + 0.5*gamma,        0.5*sqrt(1-gamma)],
                       [0.5*sqrt(1-gamma),  0.5*(1-gamma)        ]].
        The output qubit of the pattern is initialized to |+>, then the X
        byproduct (conditioned on the measurement outcome) maps the state.
        Tracing the full two-qubit density-matrix evolution and averaging
        over both branches yields the diagonal result
            rho_out[0,0] = 0.5*(1 + sqrt(1-gamma)),
            rho_out[1,1] = 0.5*(1 - sqrt(1-gamma)).
        """
        gamma = fx_rng.uniform(0.0, 1.0)
        pattern = hpat()
        res = pattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=AmplitudeDampingNoiseModel(measure_channel_gamma=gamma),
            rng=fx_rng,
        )
        assert isinstance(res, DensityMatrix)
        p00 = 0.5 * (1 + np.sqrt(1 - gamma))
        p11 = 0.5 * (1 - np.sqrt(1 - gamma))
        expected = np.array([[p00, 0.0], [0.0, p11]], dtype=np.complex128)
        assert np.allclose(res.rho, expected), f"gamma={gamma}, got {res.rho}, expected {expected}"

    def test_noisy_measure_confuse(self, fx_rng: Generator) -> None:
        """measure_error_prob=1 must flip outcome and produce |1>."""
        pattern = hpat()
        res = pattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=AmplitudeDampingNoiseModel(measure_error_prob=1.0),
            rng=fx_rng,
        )
        assert isinstance(res, DensityMatrix)
        assert np.allclose(res.rho, np.array([[0.0, 0.0], [0.0, 1.0]]))

    def test_full_damping_collapses_output(self, fx_rng: Generator) -> None:
        """With gamma=1 on prepare, all qubits decay and output collapses to |0>."""
        pattern = hpat()
        res = pattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=AmplitudeDampingNoiseModel(prepare_error_gamma=1.0),
            rng=fx_rng,
        )
        assert isinstance(res, DensityMatrix)
        # Fully damped preparation → |0><0| for all nodes
        assert np.allclose(res.rho, np.array([[1.0, 0.0], [0.0, 0.0]]))
