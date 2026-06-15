from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from graphix import Pattern
from graphix.command import CommandKind, M, N
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
from graphix.noise_models.noise_model import NoiselessNoiseModel
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


def test_amplitude_damping_noise_nqubits() -> None:
    ad_noise = AmplitudeDampingNoise(0.3)
    assert ad_noise.nqubits == 1


def test_two_qubit_amplitude_damping_noise_nqubits() -> None:
    ad2_noise = TwoQubitAmplitudeDampingNoise(0.3)
    assert ad2_noise.nqubits == 2


def test_amplitude_damping_noise_to_kraus_channel(fx_rng: Generator) -> None:
    gamma = fx_rng.uniform(0, 0.5)
    ad_noise = AmplitudeDampingNoise(gamma)
    channel = ad_noise.to_kraus_channel()
    assert channel.nqubit == 1
    assert len(channel) == 2


def test_amplitude_damping_noise_model_default() -> None:
    model = AmplitudeDampingNoiseModel()
    assert model.prepare_error_prob == 0.0
    assert model.measure_error_prob == 0.0
    assert not model.entanglement_error_prob
    assert not model.measure_channel_prob


def test_amplitude_damping_noise_model_properties(fx_rng: Generator) -> None:
    gamma = fx_rng.uniform(0, 0.5)
    model = AmplitudeDampingNoiseModel(
        prepare_error_prob=gamma,
        x_error_prob=gamma,
        z_error_prob=gamma,
        entanglement_error_prob=gamma,
        measure_channel_prob=gamma,
        measure_error_prob=gamma,
    )
    assert model.prepare_error_prob == gamma
    assert model.x_error_prob == gamma
    assert model.z_error_prob == gamma
    assert model.entanglement_error_prob == gamma
    assert model.measure_channel_prob == gamma
    assert model.measure_error_prob == gamma


def test_amplitude_damping_noise_model_input_nodes(fx_rng: Generator) -> None:
    gamma = fx_rng.uniform(0, 0.5)
    model = AmplitudeDampingNoiseModel(prepare_error_prob=gamma)
    nodes = model.input_nodes([0, 1])
    assert len(nodes) == 2
    for node_cmd in nodes:
        assert isinstance(node_cmd, ApplyNoise)
        assert isinstance(node_cmd.noise, AmplitudeDampingNoise)
        assert node_cmd.noise.gamma == gamma


def test_amplitude_damping_noise_model_confuse_result(fx_rng: Generator) -> None:
    # Pattern that measures 0 on qubit 0 with probability 1.
    pattern = Pattern(cmds=[N(0), M(0)])
    measure_method = DefaultMeasureMethod()
    pattern.simulate_pattern(
        backend="densitymatrix",
        noise_model=AmplitudeDampingNoiseModel(),
        rng=fx_rng,
        measure_method=measure_method,
    )
    assert measure_method.results[0] == 0
    noise_model = AmplitudeDampingNoiseModel(measure_error_prob=1)
    measure_method = DefaultMeasureMethod()
    pattern.simulate_pattern(
        backend="densitymatrix",
        noise_model=noise_model,
        rng=fx_rng,
        measure_method=measure_method,
    )
    assert measure_method.results[0] == 1


def test_confuse_result(fx_rng: Generator) -> None:
    # Pattern that measures 0 on qubit 0 with probability 1.
    pattern = Pattern(cmds=[N(0), M(0)])
    measure_method = DefaultMeasureMethod()
    pattern.simulate_pattern(
        backend="densitymatrix", noise_model=NoiselessNoiseModel(), rng=fx_rng, measure_method=measure_method
    )
    assert measure_method.results[0] == 0
    noise_model = DepolarisingNoiseModel(measure_error_prob=1)
    measure_method = DefaultMeasureMethod()
    pattern.simulate_pattern(
        backend="densitymatrix", noise_model=noise_model, rng=fx_rng, measure_method=measure_method
    )
    assert measure_method.results[0] == 1
