import matplotlib
import numpy as np
import pytest
from numpy.random import Generator

import graphix.command
import tests.random_circuit as rc
from graphix.parameter import Placeholder
from graphix.pattern import Pattern


def test_pattern_affine_operations() -> None:
    alpha = Placeholder("alpha")
    assert alpha + 1 + 1 == alpha + 2
    assert alpha + alpha == 2 * alpha
    assert alpha - alpha == 0
    assert alpha / 2 == 0.5 * alpha
    assert -alpha + alpha == 0
    beta = Placeholder("beta")
    with pytest.raises(graphix.parameter.PlaceholderOperationError):
        alpha + beta


def test_pattern_without_parameter_is_not_parameterized() -> None:
    pattern = Pattern(input_nodes=[0, 1])
    pattern.add(graphix.command.M(node=0))
    # A pattern without parameterized angle is not parameterized.
    assert not pattern.is_parameterized()


def test_pattern_without_parameter_subs_is_identity() -> None:
    pattern = Pattern(input_nodes=[0, 1])
    pattern.add(graphix.command.M(node=0))
    alpha = Placeholder("alpha")
    # Substitution in a pattern without parameterized angle is the identity.
    assert list(pattern) == list(pattern.subs(alpha, 0))


def test_pattern_substitution() -> None:
    pattern = Pattern(input_nodes=[0, 1])
    pattern.add(graphix.command.M(node=0))
    alpha = Placeholder("alpha")
    pattern.add(graphix.command.M(node=1, angle=alpha))
    assert pattern.is_parameterized()
    # Parameterized patterns can be substituted, even if some angles are not parameterized.
    pattern0 = pattern.subs(alpha, 0)
    # If all parameterized angles have been instantiated, the pattern is no longer parameterized.
    assert not pattern0.is_parameterized()
    assert list(pattern0) == [graphix.command.M(node=0), graphix.command.M(node=1)]


def test_instantiated_pattern_simulation() -> None:
    pattern = Pattern(input_nodes=[0, 1])
    pattern.add(graphix.command.M(node=0))
    alpha = Placeholder("alpha")
    pattern.add(graphix.command.M(node=1, angle=alpha))
    pattern0 = pattern.subs(alpha, 0)
    # Instantied patterns can be simulated.
    pattern0.simulate_pattern()
    pattern1 = pattern.subs(alpha, 1)
    assert not pattern1.is_parameterized()
    assert list(pattern1) == [graphix.command.M(node=0), graphix.command.M(node=1, angle=1)]
    pattern1.simulate_pattern()


def test_multiple_parameters() -> None:
    pattern = Pattern(input_nodes=[0, 1])
    pattern.add(graphix.command.M(node=0))
    alpha = Placeholder("alpha")
    pattern.add(graphix.command.M(node=1, angle=alpha))
    beta = Placeholder("beta")
    pattern.add(graphix.command.N(node=2))
    pattern.add(graphix.command.M(node=2, angle=beta))
    # A partially instantiated pattern is still parameterized.
    assert pattern.subs(alpha, 2).is_parameterized()
    pattern23 = pattern.subs(alpha, 2).subs(beta, 3)
    # A full instantiated pattern is no longer parameterized.
    assert not pattern23.is_parameterized()
    assert list(pattern23) == [
        graphix.command.M(node=0),
        graphix.command.M(node=1, angle=2),
        graphix.command.N(node=2),
        graphix.command.M(node=2, angle=3),
    ]
    pattern23.simulate_pattern()


@pytest.mark.parametrize("jumps", range(1, 11))
def test_random_circuit_with_parameters(fx_rng: Generator, jumps: int, use_rustworkx: bool = True) -> None:
    nqubits = 5
    depth = 5
    alpha = Placeholder("alpha")
    beta = Placeholder("beta")
    circuit = rc.get_rand_circuit(nqubits, depth, fx_rng, parameters=[alpha, beta])
    pattern = circuit.transpile().pattern
    pattern.standardize()
    pattern.shift_signals(method="global")
    pattern.perform_pauli_measurements(use_rustworkx=use_rustworkx)
    pattern.minimize_space()
    alpha_value = 2 * fx_rng.random()  # [0, 2π) / π
    beta_value = 2 * fx_rng.random()
    state = circuit.subs(alpha, alpha_value).subs(beta, beta_value).simulate_statevector().statevec
    state_mbqc = pattern.subs(alpha, alpha_value).subs(beta, beta_value).simulate_pattern()
    assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)


def test_visualization() -> None:
    matplotlib.use("Agg")  # Use a non-interactive backend
    pattern = Pattern(input_nodes=[0, 1])
    pattern.add(graphix.command.M(node=0))
    alpha = Placeholder("alpha")
    pattern.add(graphix.command.M(node=1, angle=alpha))
    pattern.draw_graph()


def test_simulation_exception() -> None:
    pattern = Pattern(input_nodes=[0, 1])
    pattern.add(graphix.command.M(node=0))
    alpha = Placeholder("alpha")
    pattern.add(graphix.command.M(node=1, angle=alpha))
    with pytest.raises(graphix.parameter.PlaceholderOperationError):
        pattern.simulate_pattern()
