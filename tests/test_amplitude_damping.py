r"""Tests for the amplitude damping channel and noise model.

Soundness of these tests
------------------------
The single-qubit amplitude damping channel has the analytic action
:math:`\\Phi_\\gamma(\\rho) = K_1 \\rho K_1^\\dagger + K_2 \\rho K_2^\\dagger`, which on
the computational basis gives :math:`\\Phi_\\gamma(|0\\rangle\\langle 0|) = |0\\rangle\\langle 0|`
(the ground state is unaffected) and
:math:`\\Phi_\\gamma(|1\\rangle\\langle 1|) = \\gamma |0\\rangle\\langle 0| + (1-\\gamma)|1\\rangle\\langle 1|`
(the excited state decays with probability ``gamma``). The two-qubit channel is the
tensor product of two independent single-qubit channels, so its action on a product
state is the Kronecker product of the single-qubit results. These closed forms are used
as references below. The noise-model tests check that the expected ``ApplyNoise`` commands
carrying amplitude damping are inserted at each step of a pattern, and that a zero-damping
model reproduces the noiseless result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from graphix import command
from graphix.channels import amplitude_damping_channel, two_qubit_amplitude_damping_channel
from graphix.command import CommandKind
from graphix.noise_models import (
    AmplitudeDampingNoise,
    AmplitudeDampingNoiseModel,
    ApplyNoise,
    TwoQubitAmplitudeDampingNoise,
)
from graphix.pattern import Pattern
from graphix.sim.density_matrix import DensityMatrix
from graphix.states import BasicStates
from graphix.transpiler import Circuit

if TYPE_CHECKING:
    from numpy.random import Generator

GAMMAS = [0.0, 0.2, 0.5, 0.9, 1.0]


def _as_apply_noise(cmd: object) -> ApplyNoise:
    assert isinstance(cmd, ApplyNoise)
    return cmd


@pytest.mark.parametrize("gamma", GAMMAS)
def test_amplitude_damping_channel_basis_states(gamma: float) -> None:
    # Ground state is unaffected.
    ground = DensityMatrix(data=[BasicStates.ZERO])
    ground.apply_channel(amplitude_damping_channel(gamma), [0])
    assert np.allclose(ground.rho, np.array([[1.0, 0.0], [0.0, 0.0]]))
    # Excited state decays to the ground state with probability gamma.
    excited = DensityMatrix(data=[BasicStates.ONE])
    excited.apply_channel(amplitude_damping_channel(gamma), [0])
    assert np.allclose(excited.rho, np.array([[gamma, 0.0], [0.0, 1.0 - gamma]]))


@pytest.mark.parametrize("gamma", GAMMAS)
def test_amplitude_damping_channel_superposition(gamma: float) -> None:
    # Analytic action on |+><+|.
    plus = DensityMatrix(data=[BasicStates.PLUS])
    plus.apply_channel(amplitude_damping_channel(gamma), [0])
    root = float(np.sqrt(1.0 - gamma))
    expected = 0.5 * np.array([[1.0 + gamma, root], [root, 1.0 - gamma]])
    assert np.allclose(plus.rho, expected)
    # Trace is preserved (CPTP).
    assert np.isclose(plus.rho.trace(), 1.0)


@pytest.mark.parametrize("gamma", GAMMAS)
def test_two_qubit_amplitude_damping_channel(gamma: float) -> None:
    # On a product state, the two-qubit channel is the Kronecker product of the
    # single-qubit actions.
    dm = DensityMatrix(data=[BasicStates.ONE, BasicStates.ONE])
    dm.apply_channel(two_qubit_amplitude_damping_channel(gamma), [0, 1])
    single = np.array([[gamma, 0.0], [0.0, 1.0 - gamma]])
    assert np.allclose(dm.rho, np.kron(single, single))


def test_channel_nqubits() -> None:
    assert amplitude_damping_channel(0.3).nqubit == 1
    assert two_qubit_amplitude_damping_channel(0.3).nqubit == 2
    assert len(amplitude_damping_channel(0.3)) == 2
    assert len(two_qubit_amplitude_damping_channel(0.3)) == 4


def test_noise_elements() -> None:
    one = AmplitudeDampingNoise(0.4)
    assert one.nqubits == 1
    assert one.to_kraus_channel().nqubit == 1
    two = TwoQubitAmplitudeDampingNoise(0.4)
    assert two.nqubits == 2
    assert two.to_kraus_channel().nqubit == 2


@pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0])
def test_noise_gamma_validation(bad: float) -> None:
    with pytest.raises(ValueError):
        AmplitudeDampingNoise(bad)


def test_noise_model_inserts_noise_at_each_step() -> None:
    model = AmplitudeDampingNoiseModel(
        prepare_error_gamma=0.1,
        x_error_gamma=0.2,
        z_error_gamma=0.3,
        entanglement_error_gamma=0.4,
        measure_channel_gamma=0.5,
    )

    # N: prepare error appended after the command.
    out = model.command(command.N(0))
    assert [c.kind for c in out] == [CommandKind.N, CommandKind.ApplyNoise]
    noise = _as_apply_noise(out[1])
    assert isinstance(noise.noise, AmplitudeDampingNoise)
    assert noise.noise.gamma == 0.1
    assert noise.nodes == [0]

    # E: two-qubit entanglement error appended after the command.
    out = model.command(command.E((0, 1)))
    assert [c.kind for c in out] == [CommandKind.E, CommandKind.ApplyNoise]
    noise = _as_apply_noise(out[1])
    assert isinstance(noise.noise, TwoQubitAmplitudeDampingNoise)
    assert noise.noise.gamma == 0.4
    assert noise.nodes == [0, 1]

    # M: measurement channel applied *before* the measurement.
    out = model.command(command.M(0))
    assert [c.kind for c in out] == [CommandKind.ApplyNoise, CommandKind.M]
    noise = _as_apply_noise(out[0])
    assert isinstance(noise.noise, AmplitudeDampingNoise)
    assert noise.noise.gamma == 0.5

    # X / Z: correction errors carry the command's domain.
    out = model.command(command.X(0, domain={1}))
    assert [c.kind for c in out] == [CommandKind.X, CommandKind.ApplyNoise]
    noise = _as_apply_noise(out[1])
    assert isinstance(noise.noise, AmplitudeDampingNoise)
    assert noise.noise.gamma == 0.2
    assert noise.domain == {1}

    out = model.command(command.Z(0, domain={1}))
    assert [c.kind for c in out] == [CommandKind.Z, CommandKind.ApplyNoise]
    noise = _as_apply_noise(out[1])
    assert isinstance(noise.noise, AmplitudeDampingNoise)
    assert noise.noise.gamma == 0.3
    assert noise.domain == {1}


def test_noise_model_input_nodes() -> None:
    model = AmplitudeDampingNoiseModel(prepare_error_gamma=0.25)
    out = model.input_nodes([0, 2])
    noises = [_as_apply_noise(c) for c in out]
    assert all(isinstance(n.noise, AmplitudeDampingNoise) and n.noise.gamma == 0.25 for n in noises)
    assert [n.nodes for n in noises] == [[0], [2]]


def test_zero_damping_is_noiseless(fx_rng: Generator) -> None:
    # A model with gamma = 0 everywhere leaves the noiseless result unchanged.
    pattern = _hadamard_pattern()
    res = pattern.simulate_pattern(backend="densitymatrix", noise_model=AmplitudeDampingNoiseModel(), rng=fx_rng)
    assert isinstance(res, DensityMatrix)
    assert np.allclose(res.rho, np.array([[1.0, 0.0], [0.0, 0.0]]))


@pytest.mark.parametrize("gamma", [0.2, 0.6])
def test_preparation_amplitude_damping(gamma: float, fx_rng: Generator) -> None:
    # A single preparation followed by amplitude damping yields the analytic
    # action of the channel on the prepared |+> state.
    pattern = Pattern(cmds=[command.N(0)])
    res = pattern.simulate_pattern(
        backend="densitymatrix", noise_model=AmplitudeDampingNoiseModel(prepare_error_gamma=gamma), rng=fx_rng
    )
    assert isinstance(res, DensityMatrix)
    root = float(np.sqrt(1.0 - gamma))
    expected = 0.5 * np.array([[1.0 + gamma, root], [root, 1.0 - gamma]])
    assert np.allclose(res.rho, expected)


def _hadamard_pattern() -> Pattern:
    circ = Circuit(1)
    circ.h(0)
    return circ.transpile().pattern
