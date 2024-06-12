import numpy as np
import pytest

from graphix.parameter import Parameter
from graphix.pattern import Pattern
from graphix.transpiler import Circuit


def test_expression() -> None:
    alpha = Parameter("alpha")
    assert str(alpha) == "alpha"
    assert str(alpha + 0) == "alpha"
    assert str(alpha + 1) == "1 + alpha"
    assert str(alpha + 1 + 1) == "2 + alpha"
    assert str(alpha.cos() + alpha.cos()) == "2 * cos(alpha)"
    assert str(alpha - alpha) == "0j"
    assert str(alpha * alpha) == "alpha ** 2"
    beta = Parameter("beta")
    assert str((alpha + beta) * (alpha - beta)) == "alpha ** 2 - beta ** 2"
    assert str((alpha + beta) ** 2) == "2 * alpha * beta + alpha ** 2 + beta ** 2"
    assert str((alpha - beta) ** 2) == "alpha ** 2 - 2 * alpha * beta + beta ** 2"


def test_parameter() -> None:
    pattern = Pattern(input_nodes=[0, 1])
    pattern.add(["M", 0, "XY", 0, [], []])
    # A pattern without parameterized angle is not parameterized.
    assert not pattern.is_parameterized()
    # Substitution in a pattern without parameterized angle is the identity.
    alpha = Parameter("alpha")
    assert list(pattern) == list(pattern.subs(alpha, 0))
    # A pattern without parameterized angle can be simulated.
    pattern.simulate_pattern()
    pattern.add(["M", 1, "XY", alpha, [], []])
    assert pattern.is_parameterized()
    # Parameterized patterns can be substituted, even if some angles are not parameterized.
    pattern0 = pattern.subs(alpha, 0)
    # If all parameterized angles have been instantiated, the pattern is no longer parameterized.
    assert not pattern0.is_parameterized()
    assert list(pattern0) == [["M", 0, "XY", 0, [], []], ["M", 1, "XY", 0, [], []]]
    # Instantied patterns can be simulated.
    pattern0.simulate_pattern()
    pattern1 = pattern.subs(alpha, 1)
    assert not pattern1.is_parameterized()
    assert list(pattern1) == [["M", 0, "XY", 0, [], []], ["M", 1, "XY", 1, [], []]]
    pattern1.simulate_pattern()
    beta = Parameter("beta")
    pattern.add(["N", 2])
    pattern.add(["M", 2, "XY", beta, [], []])
    # A partially instantiated pattern is still parameterized.
    assert pattern.subs(alpha, 2).is_parameterized()
    pattern23 = pattern.subs(alpha, 2).subs(beta, 3)
    # A full instantiated pattern is no longer parameterized.
    assert not pattern23.is_parameterized()
    assert list(pattern23) == [
        ["M", 0, "XY", 0, [], []],
        ["M", 1, "XY", 2, [], []],
        ["N", 2],
        ["M", 2, "XY", 3, [], []],
    ]
    pattern23.simulate_pattern()
    # Parameterized angles support expressions.
    pattern_beta = pattern.subs(alpha, beta + 1)
    assert pattern_beta.is_parameterized()
    # Substitution evaluates expressions.
    pattern43 = pattern_beta.subs(beta, 3)
    assert not pattern43.is_parameterized()
    assert list(pattern43) == [
        ["M", 0, "XY", 0, [], []],
        ["M", 1, "XY", 4.0, [], []],
        ["N", 2],
        ["M", 2, "XY", 3.0, [], []],
    ]
    pattern43.simulate_pattern()


def test_parameter_circuit_simulation(fx_rng: np.random.Generator) -> None:
    alpha = Parameter("alpha")
    circuit = Circuit(1)
    circuit.rz(0, alpha)
    result_subs_then_simulate = circuit.subs(alpha, 0.5).simulate_statevector().statevec
    result_simulate_then_subs = circuit.simulate_statevector().statevec.subs(alpha, 0.5)
    assert np.allclose(result_subs_then_simulate.psi, result_simulate_then_subs.psi)


@pytest.mark.parametrize("backend", ["statevector", "densitymatrix"])
def test_parameter_pattern_simulation(backend, fx_rng: np.random.Generator) -> None:
    alpha = Parameter("alpha")
    circuit = Circuit(1)
    circuit.rz(0, alpha)
    pattern = circuit.transpile().pattern
    # Both simulations (numeric and symbolic) will use the same
    # seed for random number generation, to ensure that the
    # explored branch is the same for the two simulations.
    seed = fx_rng.integers(2**63)
    result_subs_then_simulate = pattern.subs(alpha, 0.5).simulate_pattern(
        backend, pr_calc=False, rng=np.random.default_rng(seed)
    )
    # Note: pr_calc=False is mandatory since we cannot compute
    # probabilities on symbolic states; we explore one arbitrary
    # branch.
    result_simulate_then_subs = pattern.simulate_pattern(backend, pr_calc=False, rng=np.random.default_rng(seed)).subs(
        alpha, 0.5
    )
    if backend == "statevector":
        assert np.allclose(result_subs_then_simulate.psi, result_simulate_then_subs.psi)
    elif backend == "densitymatrix":
        assert np.allclose(result_subs_then_simulate.rho, result_simulate_then_subs.rho)
