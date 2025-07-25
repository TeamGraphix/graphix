from __future__ import annotations

import dataclasses
import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import pytest
from typing_extensions import override

from graphix import Pattern
from graphix.branch_selector import ConstBranchSelector, FixedBranchSelector, RandomBranchSelector
from graphix.command import M, N
from graphix.simulator import DefaultMeasureMethod

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.random import Generator

    from graphix.measurements import Outcome

NB_ROUNDS = 100


@dataclass
class CheckedBranchSelector(RandomBranchSelector):
    """Random branch selector that verifies that expectation values match the expected ones."""

    expected: Mapping[int, float] = dataclasses.field(default_factory=dict)

    @override
    def measure(self, qubit: int, compute_expectation_0: Callable[[], float]) -> Outcome:
        """Return the measurement outcome of ``qubit``."""
        expectation_0 = compute_expectation_0()
        assert math.isclose(expectation_0, self.expected[qubit])
        return super().measure(qubit, lambda: expectation_0)


@pytest.mark.filterwarnings("ignore:Simulating using densitymatrix backend with no noise.")
@pytest.mark.parametrize(
    "backend",
    [
        "statevector",
        "densitymatrix",
        pytest.param(
            "tensornetwork",
            marks=pytest.mark.xfail(
                reason="[Bug]: TensorNetworkBackend computes incorrect measurement probabilities #325"
            ),
        ),
    ],
)
def test_expectation_value(fx_rng: Generator, backend: str) -> None:
    # Pattern that measures 0 on qubit 0 with probability 1.
    pattern = Pattern(cmds=[N(0), M(0)])
    branch_selector = CheckedBranchSelector(rng=fx_rng, expected={0: 1.0})
    pattern.simulate_pattern(backend, branch_selector=branch_selector)


@pytest.mark.filterwarnings("ignore:Simulating using densitymatrix backend with no noise.")
@pytest.mark.parametrize(
    "backend",
    [
        "statevector",
        "densitymatrix",
        pytest.param(
            "tensornetwork",
            marks=pytest.mark.xfail(
                reason="[Bug]: TensorNetworkBackend computes incorrect measurement probabilities #325"
            ),
        ),
    ],
)
def test_random_branch_selector(fx_rng: Generator, backend: str) -> None:
    branch_selector = RandomBranchSelector(rng=fx_rng)
    pattern = Pattern(cmds=[N(0), M(0)])
    for _ in range(NB_ROUNDS):
        measure_method = DefaultMeasureMethod()
        pattern.simulate_pattern(backend, branch_selector=branch_selector, measure_method=measure_method)
        assert measure_method.results[0] == 0


@pytest.mark.filterwarnings("ignore:Simulating using densitymatrix backend with no noise.")
@pytest.mark.parametrize(
    "backend",
    [
        "statevector",
        "densitymatrix",
        "tensornetwork",
    ],
)
def test_random_branch_selector_without_pr_calc(backend: str) -> None:
    branch_selector = RandomBranchSelector(pr_calc=False)
    # Pattern that measures 0 on qubit 0 with probability > 0.999999999, to avoid numerical errors when exploring impossible branches.
    pattern = Pattern(cmds=[N(0), M(0, angle=1e-5)])
    nb_outcome_1 = 0
    for _ in range(NB_ROUNDS):
        measure_method = DefaultMeasureMethod()
        pattern.simulate_pattern(backend, branch_selector=branch_selector, measure_method=measure_method)
        if measure_method.results[0]:
            nb_outcome_1 += 1
    assert abs(nb_outcome_1 - NB_ROUNDS / 2) < NB_ROUNDS / 5


@pytest.mark.filterwarnings("ignore:Simulating using densitymatrix backend with no noise.")
@pytest.mark.parametrize(
    "backend",
    [
        "statevector",
        "densitymatrix",
        "tensornetwork",
    ],
)
@pytest.mark.parametrize("outcome", itertools.product([0, 1], repeat=3))
def test_fixed_branch_selector(backend: str, outcome: list[Outcome]) -> None:
    branch_selector = FixedBranchSelector(
        results=dict(enumerate(outcome[:-1])), default=FixedBranchSelector({2: outcome[2]})
    )
    pattern = Pattern(cmds=[cmd for qubit in range(3) for cmd in (N(qubit), M(qubit, angle=0.1))])
    measure_method = DefaultMeasureMethod()
    pattern.simulate_pattern(backend, branch_selector=branch_selector, measure_method=measure_method)
    for qubit, value in enumerate(outcome):
        assert measure_method.results[qubit] == value


@pytest.mark.filterwarnings("ignore:Simulating using densitymatrix backend with no noise.")
@pytest.mark.parametrize(
    "backend",
    [
        "statevector",
        "densitymatrix",
        "tensornetwork",
    ],
)
def test_fixed_branch_selector_no_default(backend: str) -> None:
    branch_selector = FixedBranchSelector(results={})
    pattern = Pattern(cmds=[N(0), M(0, angle=1e-5)])
    measure_method = DefaultMeasureMethod()
    with pytest.raises(ValueError):
        pattern.simulate_pattern(backend, branch_selector=branch_selector, measure_method=measure_method)


@pytest.mark.filterwarnings("ignore:Simulating using densitymatrix backend with no noise.")
@pytest.mark.parametrize(
    "backend",
    [
        "statevector",
        "densitymatrix",
        "tensornetwork",
    ],
)
@pytest.mark.parametrize("outcome", [0, 1])
def test_const_branch_selector(backend: str, outcome: Outcome) -> None:
    branch_selector = ConstBranchSelector(outcome)
    pattern = Pattern(cmds=[N(0), M(0, angle=1e-5)])
    for _ in range(NB_ROUNDS):
        measure_method = DefaultMeasureMethod()
        pattern.simulate_pattern(backend, branch_selector=branch_selector, measure_method=measure_method)
        assert measure_method.results[0] == outcome
