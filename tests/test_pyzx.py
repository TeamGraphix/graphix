from __future__ import annotations

import importlib.util  # Use fully-qualified import to avoid name conflict (util)
import random
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.random import PCG64, Generator

from graphix.opengraph import OpenGraph
from graphix.random_objects import rand_circuit
from graphix.transpiler import Circuit

if TYPE_CHECKING:
    from pyzx.graph.base import BaseGraph

SEED = 123


def _pyzx_notfound() -> bool:
    return importlib.util.find_spec("pyzx") is None


@pytest.mark.skipif(_pyzx_notfound(), reason="pyzx not installed")
def test_graph_equality() -> None:
    from pyzx.generate import cliffordT as clifford_t  # noqa: N813

    from graphix.pyzx import from_pyzx_graph

    random.seed(SEED)
    g = clifford_t(4, 10, 0.1)

    og1 = from_pyzx_graph(g)

    g_copy = deepcopy(g)
    og2 = from_pyzx_graph(g_copy)

    assert og1.isclose(og2)


def assert_reconstructed_pyzx_graph_equal(g: BaseGraph[int, tuple[int, int]]) -> None:
    """Convert a graph to and from an Open graph and then checks the resulting pyzx graph is equal to the original."""
    import pyzx as zx

    from graphix.pyzx import from_pyzx_graph, to_pyzx_graph

    zx.simplify.to_graph_like(g)

    g_copy = deepcopy(g)
    og = from_pyzx_graph(g_copy)
    reconstructed_pyzx_graph = to_pyzx_graph(og)

    # The "tensorfy" function break if the rows aren't set for some reason
    for v in reconstructed_pyzx_graph.vertices():
        reconstructed_pyzx_graph.set_row(v, 2)

    for v in g.vertices():
        g.set_row(v, 2)
    ten = zx.tensorfy(g)
    ten_graph = zx.tensorfy(reconstructed_pyzx_graph)
    assert zx.compare_tensors(ten, ten_graph)


# Tests that compiling from a pyzx graph to an OpenGraph returns the same
# graph. Only works with small circuits up to 4 qubits since PyZX's `tensorfy`
# function seems to consume huge amount of memory for larger qubit
@pytest.mark.skipif(_pyzx_notfound(), reason="pyzx not installed")
def test_random_clifford_t() -> None:
    from pyzx.generate import cliffordT as clifford_t  # noqa: N813

    for _ in range(15):
        g = clifford_t(4, 10, 0.1)
        assert_reconstructed_pyzx_graph_equal(g)


@pytest.mark.skipif(_pyzx_notfound(), reason="pyzx not installed")
@pytest.mark.parametrize("jumps", range(1, 11))
def test_random_circuit(fx_bg: PCG64, jumps: int) -> None:
    from graphix.pyzx import from_pyzx_graph, to_pyzx_graph

    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 5
    depth = 5
    circuit = rand_circuit(nqubits, depth, rng)
    pattern = circuit.transpile().pattern
    opengraph = OpenGraph.from_pattern(pattern)
    zx_graph = to_pyzx_graph(opengraph)
    opengraph2 = from_pyzx_graph(zx_graph)
    pattern2 = opengraph2.to_pattern()
    pattern.perform_pauli_measurements()
    pattern.minimize_space()
    state = pattern.simulate_pattern()
    pattern2.perform_pauli_measurements()
    pattern2.minimize_space()
    state2 = pattern2.simulate_pattern()
    assert np.abs(np.dot(state.flatten().conjugate(), state2.flatten())) == pytest.approx(1)


def test_rz() -> None:
    import pyzx as zx

    from graphix.pyzx import from_pyzx_graph

    circuit = Circuit(1)
    circuit.rz(0, np.pi / 4)
    pattern = circuit.transpile().pattern
    circ = zx.qasm("qreg q[1]; rz(pi/4) q[0];")
    g = circ.to_graph()
    og = from_pyzx_graph(g)
    pattern_zx = og.to_pattern()
    state = pattern.simulate_pattern()
    state_zx = pattern_zx.simulate_pattern()
    assert np.abs(np.dot(state_zx.flatten().conjugate(), state.flatten())) == pytest.approx(1)
