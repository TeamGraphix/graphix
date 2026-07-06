from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from graphix import Pattern
from graphix.clifford import Clifford
from graphix.command import C, CommandKind, E, M, N, S, T, X, Z
from graphix.noise_models import (
    AmplitudeDampingNoise,
    AmplitudeDampingNoiseModel,
    ApplyNoise,
    ComposeNoiseModel,
    DepolarisingNoise,
    DepolarisingNoiseModel,
    TwoQubitAmplitudeDampingNoise,
    TwoQubitDepolarisingNoise,
)
from graphix.noise_models.noise_model import NoiselessNoiseModel, NoiseModel
from graphix.random_objects import rand_circuit
from graphix.sim.density_matrix import DensityMatrix
from graphix.simulator import DefaultMeasureMethod

if TYPE_CHECKING:
    from numpy.random import Generator

    from graphix.noise_models import CommandOrNoise


def test_noiseless_noise_model_transpile(fx_rng: Generator) -> None:
    nqubits = 5
    depth = 5
    circuit = rand_circuit(nqubits, depth, rng=fx_rng)
    pattern = circuit.transpile().pattern
    noise_model = NoiselessNoiseModel()
    assert noise_model.transpile(pattern) == list(pattern)


def test_noiseless_noise_model_simulation(fx_rng: Generator) -> None:
    nqubits = 5
    depth = 5
    circuit = rand_circuit(nqubits, depth, rng=fx_rng)
    state = circuit.simulate_statevector().statevec
    pattern = circuit.transpile().pattern
    pattern.standardize()
    pattern.minimize_space()
    noise_model = NoiselessNoiseModel()
    state_mbqc = pattern.simulate_pattern(backend="densitymatrix", noise_model=noise_model, rng=fx_rng)
    assert np.abs(np.dot(state_mbqc.flatten().conjugate(), DensityMatrix(state).rho.flatten())) == pytest.approx(1)


def test_compose_noise_model_transpile(fx_rng: Generator) -> None:
    nqubits = 5
    depth = 5
    circuit = rand_circuit(nqubits, depth, rng=fx_rng)
    pattern = circuit.transpile().pattern
    noise_model = ComposeNoiseModel(
        [DepolarisingNoiseModel(x_error_prob=0.5), DepolarisingNoiseModel(z_error_prob=0.5)]
    )
    noisy_pattern = noise_model.transpile(pattern, rng=fx_rng)
    iterator = iter(noisy_pattern)

    def check_noise_command(cmd: CommandOrNoise, prob: float, two_qubits: bool) -> None:
        assert isinstance(cmd, ApplyNoise)
        if two_qubits:
            assert isinstance(cmd.noise, TwoQubitDepolarisingNoise)
        else:
            assert isinstance(cmd.noise, DepolarisingNoise)
        assert cmd.noise.prob == prob

    for cmd in pattern:
        if cmd.kind == CommandKind.M:
            check_noise_command(next(iterator), 0, False)
            check_noise_command(next(iterator), 0, False)
        assert next(iterator) == cmd
        match cmd.kind:
            case CommandKind.N:
                check_noise_command(next(iterator), 0, False)
                check_noise_command(next(iterator), 0, False)
            case CommandKind.E:
                check_noise_command(next(iterator), 0, True)
                check_noise_command(next(iterator), 0, True)
            case CommandKind.X:
                check_noise_command(next(iterator), 0, False)
                check_noise_command(next(iterator), 0.5, False)
            case CommandKind.Z:
                check_noise_command(next(iterator), 0.5, False)
                check_noise_command(next(iterator), 0, False)


def test_compose_noise_model_simulation(fx_rng: Generator) -> None:
    nqubits = 5
    depth = 5
    circuit = rand_circuit(nqubits, depth, rng=fx_rng)
    state = circuit.simulate_statevector().statevec
    pattern = circuit.transpile().pattern
    pattern.standardize()
    pattern.minimize_space()
    # By default, `DepolarisingNoiseModel` is noiseless.
    noise_model = ComposeNoiseModel([NoiselessNoiseModel(), DepolarisingNoiseModel()])
    state_mbqc = pattern.simulate_pattern(backend="densitymatrix", noise_model=noise_model, rng=fx_rng)
    assert np.abs(np.dot(state_mbqc.flatten().conjugate(), DensityMatrix(state).rho.flatten())) == pytest.approx(1)


def _assert_noise(
    cmd: CommandOrNoise,
    noise_type: type[AmplitudeDampingNoise | TwoQubitAmplitudeDampingNoise],
    gamma: float,
    nodes: list[int],
    domain: set[int] | None = None,
) -> None:
    assert isinstance(cmd, ApplyNoise)
    assert isinstance(cmd.noise, noise_type)
    assert cmd.noise.prob == gamma
    assert cmd.nodes == nodes
    assert cmd.domain == domain


def test_amplitude_damping_command_injection() -> None:
    """Amplitude damping noise is injected at the correct command positions."""
    model = AmplitudeDampingNoiseModel(
        prepare_error_prob=0.1,
        x_error_prob=0.2,
        z_error_prob=0.5,
        entanglement_error_prob=0.3,
        measure_channel_prob=0.4,
    )

    # N: noise applied AFTER preparation
    out = model.command(N(node=0))
    assert out[0].kind == CommandKind.N
    _assert_noise(out[1], AmplitudeDampingNoise, 0.1, [0])

    # E: two-qubit noise applied AFTER entanglement
    out = model.command(E(nodes=(0, 1)))
    assert out[0].kind == CommandKind.E
    _assert_noise(out[1], TwoQubitAmplitudeDampingNoise, 0.3, [0, 1])

    # M: noise applied BEFORE measurement
    out = model.command(M(node=0))
    _assert_noise(out[0], AmplitudeDampingNoise, 0.4, [0])
    assert out[1].kind == CommandKind.M

    # X: correction kept, noise conditioned on the same domain
    out = model.command(X(node=0, domain={1, 2}))
    assert out[0].kind == CommandKind.X
    _assert_noise(out[1], AmplitudeDampingNoise, 0.2, [0], {1, 2})

    # Z: correction kept, noise conditioned on the same domain
    out = model.command(Z(node=0, domain={3}))
    assert out[0].kind == CommandKind.Z
    _assert_noise(out[1], AmplitudeDampingNoise, 0.5, [0], {3})


@pytest.mark.parametrize(
    "noise_model",
    [DepolarisingNoiseModel(measure_error_prob=1), AmplitudeDampingNoiseModel(measure_error_prob=1)],
)
def test_confuse_result(fx_rng: Generator, noise_model: NoiseModel) -> None:
    # Pattern that measures 0 on qubit 0 with probability 1.
    pattern = Pattern(cmds=[N(0), M(0)])
    measure_method = DefaultMeasureMethod()
    pattern.simulate_pattern(
        backend="densitymatrix", noise_model=NoiselessNoiseModel(), rng=fx_rng, measure_method=measure_method
    )
    assert measure_method.results[0] == 0
    measure_method = DefaultMeasureMethod()
    pattern.simulate_pattern(
        backend="densitymatrix", noise_model=noise_model, rng=fx_rng, measure_method=measure_method
    )
    assert measure_method.results[0] == 1


def test_compose_amplitude_damping_depolarising_transpile(fx_rng: Generator) -> None:
    # Compose an amplitude damping and a depolarising model, and verify that each composed command injects the two models' noise in order.
    nqubits = 5
    depth = 5
    circuit = rand_circuit(nqubits, depth, rng=fx_rng)
    pattern = circuit.transpile().pattern
    noise_model = ComposeNoiseModel(
        [AmplitudeDampingNoiseModel(x_error_prob=0.5), DepolarisingNoiseModel(z_error_prob=0.5)]
    )
    noisy_pattern = noise_model.transpile(pattern, rng=fx_rng)
    iterator = iter(noisy_pattern)

    def check_ad(cmd: CommandOrNoise, prob: float, two_qubits: bool) -> None:
        assert isinstance(cmd, ApplyNoise)
        if two_qubits:
            assert isinstance(cmd.noise, TwoQubitAmplitudeDampingNoise)
        else:
            assert isinstance(cmd.noise, AmplitudeDampingNoise)
        assert cmd.noise.prob == prob

    def check_depol(cmd: CommandOrNoise, prob: float, two_qubits: bool) -> None:
        assert isinstance(cmd, ApplyNoise)
        if two_qubits:
            assert isinstance(cmd.noise, TwoQubitDepolarisingNoise)
        else:
            assert isinstance(cmd.noise, DepolarisingNoise)
        assert cmd.noise.prob == prob

    for cmd in pattern:
        if cmd.kind == CommandKind.M:
            # measurement-channel noise injected BEFORE M; prepend reverses the
            # order relative to appended slots, so AD precedes depol here
            check_ad(next(iterator), 0, False)
            check_depol(next(iterator), 0, False)
        assert next(iterator) == cmd
        match cmd.kind:
            case CommandKind.N:
                check_depol(next(iterator), 0, False)
                check_ad(next(iterator), 0, False)
            case CommandKind.E:
                check_depol(next(iterator), 0, True)
                check_ad(next(iterator), 0, True)
            case CommandKind.X:
                # depol carries 0 on X, AD carries x_error_prob=0.5
                check_depol(next(iterator), 0, False)
                check_ad(next(iterator), 0.5, False)
            case CommandKind.Z:
                # depol carries z_error_prob=0.5, AD carries 0 on Z
                check_depol(next(iterator), 0.5, False)
                check_ad(next(iterator), 0, False)


def test_compose_amplitude_damping_depolarising_simulation(fx_rng: Generator) -> None:
    """A composed model with both noiseless-configured models reproduces the ideal state."""
    nqubits = 5
    depth = 5
    circuit = rand_circuit(nqubits, depth, rng=fx_rng)
    state = circuit.simulate_statevector().statevec
    pattern = circuit.transpile().pattern
    pattern.standardize()
    pattern.minimize_space()
    # both models default to noiseless (all probs 0)
    noise_model = ComposeNoiseModel([AmplitudeDampingNoiseModel(), DepolarisingNoiseModel()])
    state_mbqc = pattern.simulate_pattern(backend="densitymatrix", noise_model=noise_model, rng=fx_rng)
    assert np.abs(np.dot(state_mbqc.flatten().conjugate(), DensityMatrix(state).rho.flatten())) == pytest.approx(1)


def test_amplitude_damping_nqubits() -> None:
    # covers line 46: AmplitudeDampingNoise.nqubits returning 1
    assert AmplitudeDampingNoise(0.1).nqubits == 1
    assert TwoQubitAmplitudeDampingNoise(0.1).nqubits == 2


def test_amplitude_damping_command_passthrough() -> None:
    # covers line 136: the C / T / ApplyNoise case returns [cmd] unchanged
    model = AmplitudeDampingNoiseModel()

    c_cmd = C(node=0, clifford=Clifford.H)
    assert model.command(c_cmd) == [c_cmd]

    t_cmd = T()
    assert model.command(t_cmd) == [t_cmd]

    apply_noise = ApplyNoise(noise=AmplitudeDampingNoise(0.1), nodes=[0])
    assert model.command(apply_noise) == [apply_noise]


def test_amplitude_damping_command_rejects_signal() -> None:
    # covers lines 138-139: the CommandKind.S case raises ValueError
    model = AmplitudeDampingNoiseModel()
    s_cmd = S(node=0)
    with pytest.raises(ValueError, match="Unexpected signal"):
        model.command(s_cmd)
