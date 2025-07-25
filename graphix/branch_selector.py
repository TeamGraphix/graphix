"""Branch selector.

Branch selectors determine the computation branch that is explored
during a simulation, meaning the choice of measurement outcomes.  The
branch selection can be random (see :class:`RandomBranchSelector`) or
deterministic (see :class:`ConstBranchSelector`).

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from typing_extensions import override

from graphix.measurements import outcome
from graphix.rng import ensure_rng

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.random import Generator

    from graphix.measurements import Outcome


class BranchSelector(ABC):
    """Abstract class for branch selectors.

    A branch selector provides the method `measure`, which returns the
    measurement outcome (0 or 1) for a given qubit.
    """

    @abstractmethod
    def measure(self, qubit: int, compute_expectation_0: Callable[[], float]) -> Outcome:
        """Return the measurement outcome of ``qubit``.

        Parameters
        ----------
        qubit : int
            Index of qubit to measure

        compute_expectation_0 : Callable[[], float]
            A function that the method can use to retrieve the expected
            probability of outcome 0. The probability is computed only if
            this function is called (lazy computation), ensuring no
            unnecessary computational cost.
        """


@dataclass
class RandomBranchSelector(BranchSelector):
    """Random branch selector.

    Parameters
    ----------
    pr_calc : bool, optional
        Whether to compute the probability distribution before selecting the measurement result.
        If ``False``, measurements yield 0/1 with equal probability (50% each).
        Default is ``True``.
    rng : Generator | None, optional
        Random-number generator for measurements.
        If ``None``, a default random-number generator is used.
        Default is ``None``.
    """

    pr_calc: bool = True
    rng: Generator | None = None

    @override
    def measure(self, qubit: int, compute_expectation_0: Callable[[], float]) -> Outcome:
        """
        Return the measurement outcome of ``qubit``.

        If ``pr_calc`` is ``True``, the measurement outcome is determined based on the
        computed probability of outcome 0. Otherwise, the result is randomly chosen
        with a 50% chance for either outcome.
        """
        self.rng = ensure_rng(self.rng)
        if self.pr_calc:
            prob_0 = compute_expectation_0()
            return outcome(self.rng.random() > prob_0)
        result: Outcome = self.rng.choice([0, 1])
        return result


@dataclass
class FixedBranchSelector(BranchSelector):
    """Branch selector with predefined measurement outcomes.

    The mapping is fixed in ``results``. By default, an error is raised if
    a qubit is measured without a predefined outcome. However, another
    branch selector can be specified in ``default`` to handle such cases.

    Parameters
    ----------
    results : Mapping[int, bool]
        A dictionary mapping qubits to their measurement outcomes.
        If a qubit is not present in this mapping, the ``default`` branch
        selector is used.
    default : BranchSelector | None, optional
        Branch selector to use for qubits not present in ``results``.
        If ``None``, an error is raised when an unmapped qubit is measured.
        Default is ``None``.
    """

    results: Mapping[int, Outcome]
    default: BranchSelector | None = None

    @override
    def measure(self, qubit: int, compute_expectation_0: Callable[[], float]) -> Outcome:
        """
        Return the predefined measurement outcome of ``qubit``, if available.

        If the qubit is not present in ``results``, the ``default`` branch selector
        is used. If no default is provided, an error is raised.
        """
        result = self.results.get(qubit)
        if result is None:
            if self.default is None:
                raise ValueError(f"Unexpected measurement of qubit {qubit}.")
            return self.default.measure(qubit, compute_expectation_0)
        return result


@dataclass
class ConstBranchSelector(BranchSelector):
    """Branch selector with a constant measurement outcome.

    The value ``result`` is returned for every qubit.

    Parameters
    ----------
    result : Outcome
        The fixed measurement outcome for all qubits.
    """

    result: Outcome

    @override
    def measure(self, qubit: int, compute_expectation_0: Callable[[], float]) -> Outcome:
        """Return the constant measurement outcome ``result`` for any qubit."""
        return self.result
