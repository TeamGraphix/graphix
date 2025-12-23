"""Benchmark `graphix.pattern.Pattern` methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarks.common import fx_rng
from graphix.random_objects import rand_circuit

if TYPE_CHECKING:
    from numpy.random import Generator

    from graphix.pattern import Pattern


def random_pattern(nqubits: int, depth: int, rng: Generator) -> Pattern:
    """Generate a random pattern from a random circuit.

    Parameters
    ----------
    nqubits : int
    depth : int

    Returns
    -------
    Pattern
    """
    circuit = rand_circuit(nqubits, depth, rng)
    pattern = circuit.transpile().pattern
    pattern.standardize()
    pattern.minimize_space()
    return pattern


class PatternSimulation:
    """Benchmark pattern simulation with various backends."""

    def setup(self) -> None:
        """Set up benchmark suit."""
        nqubits = 3
        depth = 5
        ncircuits = 3
        self.patterns = [random_pattern(nqubits, depth, fx_rng()) for _ in range(ncircuits)]

    def time_statevector(self) -> None:
        """Time statevector backend."""
        for pattern in self.patterns:
            pattern.simulate_pattern(backend='statevector')

    def time_densitymatrix(self) -> None:
        """Time density matrix backend."""
        for pattern in self.patterns:
            pattern.simulate_pattern(backend='densitymatrix')
