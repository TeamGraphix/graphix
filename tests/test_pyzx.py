from __future__ import annotations

import random
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.random import PCG64, Generator

from graphix.fundamentals import ANGLE_PI
from graphix.random_objects import rand_circuit
from graphix.transpiler import Circuit

try:
    import pyzx as zx
    from pyzx.generate import cliffordT as clifford_t  # noqa: N813

    from graphix.pyzx import from_pyzx_graph, to_pyzx_graph
except ImportError:
    pytestmark = pytest.mark.skip(reason="pyzx not installed")

    if TYPE_CHECKING:
        import sys

        # We skip type-checking the case where there is no pyzx, since
        # pyright cannot figure out that tests are skipped in this
        # case.
        sys.exit(1)


if TYPE_CHECKING:
    from pyzx.graph.base import BaseGraph
SEED = 123


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
    reconstructed_pyzx_graph = to_pyzx_graph(og.to_bloch())

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
def test_random_clifford_t() -> None:
    for _ in range(15):
        g = clifford_t(4, 10, 0.1)
        assert_reconstructed_pyzx_graph_equal(g)


@pytest.mark.parametrize("jumps", range(1, 11))
def test_random_circuit(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 5
    depth = 5
    circuit = rand_circuit(nqubits, depth, rng)
    pattern = circuit.transpile().pattern
    opengraph = pattern.extract_opengraph()
    zx_graph = to_pyzx_graph(opengraph.to_bloch())
    opengraph2 = from_pyzx_graph(zx_graph)
    pattern2 = opengraph2.to_pattern().infer_pauli_measurements()
    pattern.remove_input_nodes()
    pattern.perform_pauli_measurements()
    pattern.minimize_space()
    state = pattern.simulate_pattern()
    pattern2.remove_input_nodes()
    pattern2.perform_pauli_measurements()
    pattern2.minimize_space()
    state2 = pattern2.simulate_pattern()
    assert state.isclose(state2)


def test_rz() -> None:
    circuit = Circuit(2)
    circuit.rz(0, ANGLE_PI / 4)
    pattern = circuit.transpile().pattern
    # pyzx 0.8 does not support arithmetic expressions such as `pi / 4`.
    circ = zx.qasm(f"qreg q[2]; rz({np.pi / 4}) q[0];")  # type: ignore[attr-defined]
    g = circ.to_graph()
    og = from_pyzx_graph(g).infer_pauli_measurements()
    pattern_zx = og.to_pattern()
    state = pattern.simulate_pattern()
    state_zx = pattern_zx.simulate_pattern()
    assert state_zx.isclose(state)


# Issue #235
def test_full_reduce_toffoli() -> None:
    c = Circuit(3)
    c.ccx(0, 1, 2)
    p = c.transpile().pattern
    og = p.extract_opengraph()
    pyg = to_pyzx_graph(og.to_bloch())
    pyg.normalize()
    pyg_copy = deepcopy(pyg)
    zx.simplify.full_reduce(pyg)
    pyg.normalize()
    t = zx.tensorfy(pyg)
    t2 = zx.tensorfy(pyg_copy)
    assert zx.compare_tensors(t, t2)
    og2 = from_pyzx_graph(pyg).infer_pauli_measurements()
    p2 = og2.to_pattern()
    _ = p.simulate_pattern()
    _ = p2.simulate_pattern()
