from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import typing_extensions

from graphix import Pattern
from graphix.command import CommandKind, M, N, S, T
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


def test_amplitude_damping_noise_model_transpile(fx_rng: Generator) -> None:
    noise_model = AmplitudeDampingNoiseModel(
        prepare_error_prob=0.1,
        entanglement_error_prob=0.2,
        measure_channel_prob=0.3,
        x_error_prob=0.4,
        z_error_prob=0.5,
    )
    nqubits = 5
    depth = 5
    pattern = rand_circuit(nqubits, depth, rng=fx_rng).transpile().pattern
    noisy_pattern = noise_model.transpile(pattern, rng=fx_rng)
    iterator = iter(noisy_pattern)

    for cmd in pattern:
        if cmd.kind == CommandKind.M:
            assert_apply_noise(next(iterator), AmplitudeDampingNoise, 0.3, [cmd.node])
        assert next(iterator) == cmd
        match cmd.kind:
            case CommandKind.N:
                assert_apply_noise(next(iterator), AmplitudeDampingNoise, 0.1, [cmd.node])
            case CommandKind.E:
                assert_apply_noise(next(iterator), TwoQubitAmplitudeDampingNoise, 0.2, list(cmd.nodes))
            case CommandKind.X:
                assert_apply_noise(next(iterator), AmplitudeDampingNoise, 0.4, [cmd.node], cmd.domain)
            case CommandKind.Z:
                assert_apply_noise(next(iterator), AmplitudeDampingNoise, 0.5, [cmd.node], cmd.domain)
            case CommandKind.C | CommandKind.M | CommandKind.T:
                pass
            case CommandKind.S:
                raise AssertionError("Unexpected signal in pattern")
            case _:
                typing_extensions.assert_never(cmd.kind)


def test_amplitude_damping_noise_model_command_edge_cases() -> None:
    noise_model = AmplitudeDampingNoiseModel()
    apply_noise = ApplyNoise(noise=AmplitudeDampingNoise(0.2), nodes=[0])
    tick = T()

    assert noise_model.command(tick) == [tick]
    assert noise_model.command(apply_noise) == [apply_noise]
    with pytest.raises(ValueError, match="Unexpected signal"):
        noise_model.command(S(0))


def test_amplitude_damping_noise_model_input_nodes() -> None:
    noise_model = AmplitudeDampingNoiseModel(prepare_error_prob=0.25)
    cmds = noise_model.input_nodes([0, 1])
    assert len(cmds) == 2
    assert_apply_noise(cmds[0], AmplitudeDampingNoise, 0.25, [0])
    assert_apply_noise(cmds[1], AmplitudeDampingNoise, 0.25, [1])


def test_amplitude_damping_noise_nqubits() -> None:
    assert AmplitudeDampingNoise(0.3).nqubits == 1
    assert TwoQubitAmplitudeDampingNoise(0.3).nqubits == 2


def test_amplitude_damping_noise_to_kraus_channel(fx_rng: Generator) -> None:
    gamma = fx_rng.uniform()
    channel = AmplitudeDampingNoise(gamma).to_kraus_channel()
    assert channel.nqubit == 1
    assert len(channel) == 2


def test_two_qubit_amplitude_damping_noise_to_kraus_channel(fx_rng: Generator) -> None:
    gamma = fx_rng.uniform()
    channel = TwoQubitAmplitudeDampingNoise(gamma).to_kraus_channel()
    assert channel.nqubit == 2
    assert len(channel) == 4


def test_amplitude_damping_confuse_result_unchanged(fx_rng: Generator) -> None:
    noise_model = AmplitudeDampingNoiseModel(measure_error_prob=0.0)
    assert noise_model.confuse_result(M(0), 0, rng=fx_rng) == 0
    assert noise_model.confuse_result(M(0), 1, rng=fx_rng) == 1


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


def assert_apply_noise(
    cmd: CommandOrNoise,
    noise_type: type[AmplitudeDampingNoise | TwoQubitAmplitudeDampingNoise],
    gamma: float,
    nodes: list[int],
    domain: set[int] | None = None,
) -> None:
    assert isinstance(cmd, ApplyNoise)
    assert isinstance(cmd.noise, noise_type)
    assert cmd.noise.gamma == gamma
    assert cmd.nodes == nodes
    assert cmd.domain == domain
