"""Benchmark `graphix.opengraph.OpenGraph` methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarks.common import fx_rng
from graphix.random_objects import rand_circuit

if TYPE_CHECKING:
    from numpy.random import Generator

    from graphix.measurements import Measurement
    from graphix.opengraph import OpenGraph


def og_from_rnd_circuit(nqubits: int, depth: int, rng: Generator) -> OpenGraph[Measurement]:
    """Generate an open graph from a random circuit.

    Parameters
    ----------
    nqubits : int
    depth : int

    Returns
    -------
    Pattern
    """
    circuit = rand_circuit(nqubits, depth, rng)
    return circuit.transpile().pattern.extract_opengraph()


class FlowExtraction:
    """Benchmark flow extraction."""

    def setup(self) -> None:
        """Set up benchmark suit."""
        nqubits = 10
        depth = 5
        ncircuits = 5
        self.og_rnd_circuit = [og_from_rnd_circuit(nqubits, depth, fx_rng()) for _ in range(ncircuits)]

    def time_causal_flow_rnd_circuit(self) -> None:
        """Time causal-flow finding algorithm on open graphs generated from random circuits."""
        for og in self.og_rnd_circuit:
            og.find_causal_flow()

    def time_gflow_rnd_circuit(self) -> None:
        """Time generalised-flow (gflow) finding algorithm on open graphs generated from random circuits."""
        for og in self.og_rnd_circuit:
            og.find_gflow()

    def time_pauli_flow_rnd_circuit(self) -> None:
        """Time Pauli-flow finding algorithm on open graphs generated from random circuits."""
        for og in self.og_rnd_circuit:
            og.extract_pauli_flow()
