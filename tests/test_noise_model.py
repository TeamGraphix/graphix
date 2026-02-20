from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from graphix import Pattern
from graphix.command import CommandKind, M, N
from graphix.noise_models import (
    ApplyNoise,
    CommandOrNoise,
    ComposeNoiseModel,
    DepolarisingNoise,
    DepolarisingNoiseModel,
    TwoQubitDepolarisingNoise,
)
from graphix.noise_models.noise_model import NoiselessNoiseModel
from graphix.random_objects import rand_circuit
from graphix.sim.density_matrix import DensityMatrix
from graphix.simulator import DefaultMeasureMethod

if TYPE_CHECKING:
    from numpy.random import Generator


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
    noisy_pattern = noise_model.transpile(pattern)
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
