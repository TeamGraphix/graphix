import importlib.util
from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np
import pytest
from numpy.random import Generator

import graphix
import graphix.command
from graphix.device_interface import PatternRunner
from graphix.parameter import Placeholder, PlaceholderOperationError
from graphix.pattern import Pattern
from graphix.random_objects import rand_circuit
from graphix.sim.density_matrix import DensityMatrix
from graphix.sim.statevec import Statevec

if TYPE_CHECKING:
    from graphix.parameter import Parameter


def test_pattern_affine_operations() -> None:
    alpha = Placeholder("alpha")
    assert alpha + 1 + 1 == alpha + 2
    assert alpha + alpha == 2 * alpha
    assert alpha - alpha == 0
    assert alpha / 2 == 0.5 * alpha
    assert -alpha + alpha == 0
    beta = Placeholder("beta")
    with pytest.raises(PlaceholderOperationError):
        _ = alpha + beta


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


def test_parallel_substitution() -> None:
    pattern = Pattern(input_nodes=[0, 1])
    pattern.add(graphix.command.M(node=0))
    alpha = Placeholder("alpha")
    pattern.add(graphix.command.M(node=1, angle=alpha))
    beta = Placeholder("beta")
    pattern.add(graphix.command.N(node=2))
    pattern.add(graphix.command.M(node=2, angle=beta))
    pattern23 = pattern.xreplace({alpha: 2, beta: 3})
    assert not pattern23.is_parameterized()


def test_parallel_substitution_with_zero() -> None:
    # To catch potential 0 / None confusion
    pattern = Pattern(input_nodes=[0, 1])
    pattern.add(graphix.command.M(node=0))
    alpha = Placeholder("alpha")
    pattern.add(graphix.command.M(node=1, angle=alpha))
    beta = Placeholder("beta")
    pattern.add(graphix.command.N(node=2))
    pattern.add(graphix.command.M(node=2, angle=beta))
    pattern23 = pattern.xreplace({alpha: 0, beta: 0})
    assert not pattern23.is_parameterized()


def test_statevec_subs() -> None:
    alpha = Placeholder("alpha")
    statevec = Statevec([alpha])
    assert np.allclose(statevec.subs(alpha, 1).psi, np.array([1]))


def test_statevec_xreplace() -> None:
    alpha = Placeholder("alpha")
    beta = Placeholder("beta")
    statevec = Statevec([alpha, beta])
    assert np.allclose(statevec.xreplace({alpha: 1, beta: 2}).psi, np.array([1, 2]))


def test_density_matrix_subs() -> None:
    alpha = Placeholder("alpha")
    dm = DensityMatrix([[alpha]])
    assert np.allclose(dm.subs(alpha, 1).rho, np.array([1]))


def test_density_matrix_xreplace() -> None:
    alpha = Placeholder("alpha")
    beta = Placeholder("beta")
    dm = DensityMatrix([[alpha, beta], [alpha, beta]])
    assert np.allclose(dm.xreplace({alpha: 1, beta: 2}).rho, np.array([[1, 2], [1, 2]]))


@pytest.mark.parametrize("jumps", range(1, 11))
@pytest.mark.parametrize("use_xreplace", [False, True])
def test_random_circuit_with_parameters(
    fx_rng: Generator, jumps: int, use_xreplace: bool, use_rustworkx: bool = True
) -> None:
    nqubits = 5
    depth = 5
    alpha = Placeholder("alpha")
    beta = Placeholder("beta")
    circuit = rand_circuit(nqubits, depth, fx_rng, parameters=[alpha, beta])
    pattern = circuit.transpile().pattern
    pattern.standardize()
    pattern.shift_signals()
    pattern.perform_pauli_measurements(use_rustworkx=use_rustworkx)
    pattern.minimize_space()
    assignment: dict[Parameter, float] = {alpha: fx_rng.uniform(high=2), beta: fx_rng.uniform(high=2)}
    if use_xreplace:
        state = circuit.xreplace(assignment).simulate_statevector().statevec
        state_mbqc = pattern.xreplace(assignment).simulate_pattern()
    else:
        state = circuit.subs(alpha, assignment[alpha]).subs(beta, assignment[beta]).simulate_statevector().statevec
        state_mbqc = pattern.subs(alpha, assignment[alpha]).subs(beta, assignment[beta]).simulate_pattern()
    assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)


def test_visualization() -> None:
    mpl.use("Agg")  # Use a non-interactive backend
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
    with pytest.raises(PlaceholderOperationError):
        pattern.simulate_pattern()


@pytest.mark.skipif(
    importlib.util.find_spec("qiskit") is None or importlib.util.find_spec("graphix_ibmq") is None,
    reason="qiskit and/or graphix-ibmq not installed",
)
def test_ibmq_backend() -> None:
    import qiskit.circuit.exceptions

    circuit = graphix.Circuit(1)
    alpha = Placeholder("alpha")
    circuit.rx(0, alpha)
    pattern = circuit.transpile().pattern
    with pytest.raises(qiskit.circuit.exceptions.CircuitError):
        # Invalid param type <class 'graphix.parameter.AffineExpression'> for gate p.
        PatternRunner(pattern, backend="ibmq", save_statevector=True)
