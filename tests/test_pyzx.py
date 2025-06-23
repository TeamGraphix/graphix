from __future__ import annotations

import random
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.random import PCG64, Generator

from graphix.opengraph import OpenGraph
from graphix.random_objects import rand_circuit
from graphix.transpiler import Circuit

try:
    import pyzx as zx
    from pyzx.generate import cliffordT as clifford_t  # noqa: N813

    from graphix.pyzx import from_pyzx_graph, to_pyzx_graph

    _HAS_PYZX = True
except ImportError:
    _HAS_PYZX = False

if TYPE_CHECKING:
    from pyzx.graph.base import BaseGraph

SEED = 123


@pytest.mark.skipif(not _HAS_PYZX, reason="pyzx not installed")
def test_graph_equality() -> None:
    random.seed(SEED)
    g = clifford_t(4, 10, 0.1)

    og1 = from_pyzx_graph(g)

    g_copy = deepcopy(g)
    og2 = from_pyzx_graph(g_copy)

    assert og1.isclose(og2)


def assert_reconstructed_pyzx_graph_equal(g: BaseGraph[int, tuple[int, int]]) -> None:
    """Convert a graph to and from an Open graph and then checks the resulting pyzx graph is equal to the original."""
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
@pytest.mark.skipif(not _HAS_PYZX, reason="pyzx not installed")
def test_random_clifford_t() -> None:
    for _ in range(15):
        g = clifford_t(4, 10, 0.1)
        assert_reconstructed_pyzx_graph_equal(g)


@pytest.mark.skipif(not _HAS_PYZX, reason="pyzx not installed")
@pytest.mark.parametrize("jumps", range(1, 11))
def test_random_circuit(fx_bg: PCG64, jumps: int) -> None:
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


@pytest.mark.skipif(not _HAS_PYZX, reason="pyzx not installed")
def test_rz() -> None:
    circuit = Circuit(2)
    circuit.rz(0, np.pi / 4)
    pattern = circuit.transpile().pattern
    circ = zx.qasm("qreg q[2]; rz(pi / 4) q[0];")  # type: ignore[attr-defined]
    g = circ.to_graph()
    og = from_pyzx_graph(g)
    pattern_zx = og.to_pattern()
    state = pattern.simulate_pattern()
    state_zx = pattern_zx.simulate_pattern()
    assert np.abs(np.dot(state_zx.flatten().conjugate(), state.flatten())) == pytest.approx(1)


# Issue #235
@pytest.mark.skipif(not _HAS_PYZX, reason="pyzx not installed")
def test_full_reduce_toffoli() -> None:
    c = Circuit(3)
    c.ccx(0, 1, 2)
    p = c.transpile().pattern
    og = OpenGraph.from_pattern(p)
    pyg = to_pyzx_graph(og)
    pyg.normalize()
    pyg_copy = deepcopy(pyg)
    zx.simplify.full_reduce(pyg)
    pyg.normalize()
    t = zx.tensorfy(pyg)
    t2 = zx.tensorfy(pyg_copy)
    assert zx.compare_tensors(t, t2)
    og2 = from_pyzx_graph(pyg)
    p2 = og2.to_pattern()
    s = p.simulate_pattern()
    s2 = p2.simulate_pattern()
    print(np.abs(np.dot(s.flatten().conj(), s2.flatten())))
