from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import networkx as nx
import numpy as np
import pytest

# override introduced in Python 3.12
from typing_extensions import override

import graphix.command
from graphix import OpenGraph
from graphix.measurements import Measurement
from graphix.parameter import Placeholder, PlaceholderOperationError
from graphix.pattern import DrawPatternAnnotations, Pattern
from graphix.random_objects import rand_circuit
from graphix.sim.density_matrix import DensityMatrix
from graphix.sim.statevec import AbstractStatevector

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt
    from numpy.random import PCG64, Generator

    from graphix.parameter import Parameter
    from graphix.sim.base_backend import Matrix
    from graphix.sim.data import Data


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
    assert list(pattern) == list(pattern.with_parameter(alpha, 0))


def test_pattern_substitution() -> None:
    pattern = Pattern(input_nodes=[0, 1])
    pattern.add(graphix.command.M(0))
    alpha = Placeholder("alpha")
    pattern.add(graphix.command.M(1, Measurement.XY(alpha)))
    assert pattern.is_parameterized()
    # Parameterized patterns can be substituted, even if some angles are not parameterized.
    pattern0 = pattern.with_parameter(alpha, 0)
    # If all parameterized angles have been instantiated, the pattern is no longer parameterized.
    assert not pattern0.is_parameterized()
    assert list(pattern0) == [graphix.command.M(0), graphix.command.M(1, Measurement.XY(0))]
    assert list(pattern0.infer_pauli_measurements()) == [graphix.command.M(0), graphix.command.M(1)]


def test_placeholder_with_parameter_subs() -> None:
    alpha = Placeholder("alpha")
    assert alpha.with_parameter(alpha, 1) == 1
    assert alpha.subs(alpha, 1) == 1


def test_placeholder_with_parameters_xreplace() -> None:
    alpha = Placeholder("alpha")
    assert alpha.with_parameters({alpha: 1}) == 1
    assert alpha.xreplace({alpha: 1}) == 1


def test_opengraph_with_parameter_subs() -> None:
    alpha = Placeholder("alpha")
    og = OpenGraph(graph=nx.Graph([(0, 1)]), input_nodes=[0], output_nodes=[1], measurements={0: Measurement.XY(alpha)})
    og1 = og.with_parameter(alpha, 0.25)
    og2 = og.subs(alpha, 0.25)
    assert og.measurements == {0: Measurement.XY(alpha)}
    assert og1.measurements == {0: Measurement.XY(0.25)}
    assert og2.measurements == {0: Measurement.XY(0.25)}


def test_opengraph_with_parameters_xreplace() -> None:
    alpha = Placeholder("alpha")
    beta = Placeholder("beta")
    og = OpenGraph(
        graph=nx.path_graph(3),
        input_nodes=[0],
        output_nodes=[2],
        measurements={0: Measurement.XY(alpha), 1: Measurement.XY(beta)},
    )
    og1 = og.with_parameters({alpha: beta, beta: alpha})
    og2 = og.xreplace({alpha: beta, beta: alpha})
    assert og.measurements == {0: Measurement.XY(alpha), 1: Measurement.XY(beta)}
    assert og1.measurements == {0: Measurement.XY(beta), 1: Measurement.XY(alpha)}
    assert og2.measurements == {0: Measurement.XY(beta), 1: Measurement.XY(alpha)}


def test_instantiated_pattern_simulation(fx_rng: Generator) -> None:
    pattern = Pattern(input_nodes=[0, 1])
    pattern.add(graphix.command.M(node=0))
    alpha = Placeholder("alpha")
    pattern.add(graphix.command.M(1, Measurement.XY(alpha)))
    pattern0 = pattern.with_parameter(alpha, 0)
    # Instantied patterns can be simulated.
    pattern0.simulate(rng=fx_rng)
    pattern1 = pattern.with_parameter(alpha, 1)
    assert not pattern1.is_parameterized()
    assert list(pattern1) == [graphix.command.M(node=0), graphix.command.M(1, Measurement.XY(1))]
    assert list(pattern1.infer_pauli_measurements()) == [
        graphix.command.M(node=0),
        graphix.command.M(1, -Measurement.X),
    ]
    pattern1.simulate(rng=fx_rng)


def test_multiple_parameters(fx_rng: Generator) -> None:
    pattern = Pattern(input_nodes=[0, 1])
    pattern.add(graphix.command.M(node=0))
    alpha = Placeholder("alpha")
    pattern.add(graphix.command.M(1, Measurement.XY(alpha)))
    beta = Placeholder("beta")
    pattern.add(graphix.command.N(node=2))
    pattern.add(graphix.command.M(2, Measurement.XY(beta)))
    pattern.replace_parameter(alpha, 2)
    # A partially instantiated pattern is still parameterized.
    assert pattern.is_parameterized()
    pattern23 = pattern.subs(beta, 3)
    # A full instantiated pattern is no longer parameterized.
    assert not pattern23.is_parameterized()
    assert list(pattern23) == [
        graphix.command.M(node=0),
        graphix.command.M(1, Measurement.XY(2)),
        graphix.command.N(node=2),
        graphix.command.M(2, Measurement.XY(3)),
    ]
    assert list(pattern23.infer_pauli_measurements()) == [
        graphix.command.M(node=0),
        graphix.command.M(1, Measurement.X),
        graphix.command.N(node=2),
        graphix.command.M(2, -Measurement.X),
    ]
    pattern23.simulate(rng=fx_rng)


def test_parallel_substitution() -> None:
    pattern = Pattern(input_nodes=[0, 1])
    pattern.add(graphix.command.M(node=0))
    alpha = Placeholder("alpha")
    pattern.add(graphix.command.M(1, Measurement.XY(alpha)))
    beta = Placeholder("beta")
    pattern.add(graphix.command.N(node=2))
    pattern.add(graphix.command.M(2, Measurement.XY(beta)))
    pattern23 = pattern.xreplace({alpha: 2, beta: 3})
    assert not pattern23.is_parameterized()


def test_parallel_substitution_with_zero() -> None:
    # To catch potential 0 / None confusion
    pattern = Pattern(input_nodes=[0, 1])
    pattern.add(graphix.command.M(node=0))
    alpha = Placeholder("alpha")
    pattern.add(graphix.command.M(1, Measurement.XY(alpha)))
    beta = Placeholder("beta")
    pattern.add(graphix.command.N(node=2))
    pattern.add(graphix.command.M(2, Measurement.XY(beta)))
    pattern23 = pattern.xreplace({alpha: 0, beta: 0})
    assert not pattern23.is_parameterized()


def test_statevector_flatten() -> None:
    alpha = Placeholder("alpha")

    class PlaceholderStatevector(AbstractStatevector[np.object_]):
        @property
        @override
        def psi(self) -> npt.NDArray[np.object_]:
            return np.array([alpha])

        @override
        def add_nodes(self, nqubit: int, data: Data) -> None:
            raise NotImplementedError

        @override
        def entangle(self, qubits: tuple[int, int]) -> None:
            raise NotImplementedError

        @override
        def evolve(self, op: Matrix, qubits: Sequence[int]) -> None:
            raise NotImplementedError

        @override
        def evolve_single(self, op: Matrix, qubit: int) -> None:
            raise NotImplementedError

        @override
        def expectation_single(self, op: Matrix, qubit: int) -> complex:
            raise NotImplementedError

        @override
        def remove_qubit(self, qubit: int) -> None:
            raise NotImplementedError

        @override
        def swap(self, qubits: tuple[int, int]) -> None:
            raise NotImplementedError

        @override
        def permute(self, permutation: Sequence[int]) -> None:
            raise NotImplementedError

        # Note that `@property` must appear before `@override` for pyright
        @property
        @override
        def nqubit(self) -> int:
            return 1

    statevec = PlaceholderStatevector()
    assert statevec.flatten()[0] == alpha


def test_density_matrix_with_parameter() -> None:
    alpha = Placeholder("alpha")
    dm = DensityMatrix([[alpha]])
    assert np.allclose(dm.with_parameter(alpha, 1).rho, np.array([1]))
    with pytest.raises(TypeError):
        np.allclose(dm.rho, np.array([1]))
    dm.replace_parameter(alpha, 1)
    assert np.allclose(dm.rho, np.array([1]))


def test_density_matrix_with_parameters() -> None:
    alpha = Placeholder("alpha")
    beta = Placeholder("beta")
    dm = DensityMatrix([[alpha, beta], [alpha, beta]])
    assert np.allclose(dm.with_parameters({alpha: 1, beta: 2}).rho, np.array([[1, 2], [1, 2]]))
    with pytest.raises(TypeError):
        np.allclose(dm.rho, np.array([[1, 2], [1, 2]]))
    dm.replace_parameters({alpha: 1, beta: 2})
    assert np.allclose(dm.rho, np.array([[1, 2], [1, 2]]))


@pytest.mark.parametrize("jumps", range(1, 11))
@pytest.mark.parametrize("use_parallel", [False, True])
def test_random_circuit_with_parameters(fx_bg: PCG64, jumps: int, use_parallel: bool) -> None:
    rng = np.random.Generator(fx_bg.jumped(jumps))
    nqubits = 5
    depth = 5
    alpha = Placeholder("alpha")
    beta = Placeholder("beta")
    circuit = rand_circuit(nqubits, depth, rng, parameters=[alpha, beta])
    pattern = circuit.transpile().pattern
    pattern.standardize()
    pattern.shift_signals()
    pattern.infer_pauli_measurements()
    pattern.remove_pauli_measurements()
    pattern.minimize_space()
    assignment: dict[Parameter, float] = {alpha: rng.uniform(high=2), beta: rng.uniform(high=2)}
    if use_parallel:
        circuit.replace_parameters(assignment)
        pattern.replace_parameters(assignment)
    else:
        circuit.replace_parameter(alpha, assignment[alpha])
        circuit.replace_parameter(beta, assignment[beta])
        pattern.replace_parameter(alpha, assignment[alpha])
        pattern.replace_parameter(beta, assignment[beta])
    state = circuit.simulate().state
    state_mbqc = pattern.simulate(rng=rng)
    assert state_mbqc.isclose(state)


@pytest.mark.usefixtures("mock_plot")
def test_visualization() -> None:
    mpl.use("Agg")  # Use a non-interactive backend
    pattern = Pattern(input_nodes=[0, 1])
    pattern.add(graphix.command.M(node=0))
    alpha = Placeholder("alpha")
    pattern.add(graphix.command.M(1, Measurement.XY(alpha)))
    pattern.draw(annotations=DrawPatternAnnotations.XZCorrections)


def test_simulation_exception(fx_rng: Generator) -> None:
    pattern = Pattern(input_nodes=[0, 1])
    pattern.add(graphix.command.M(node=0))
    alpha = Placeholder("alpha")
    pattern.add(graphix.command.M(1, Measurement.XY(alpha)))
    with pytest.raises(PlaceholderOperationError):
        pattern.simulate(rng=fx_rng)
