from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.random import PCG64, Generator

from graphix.fundamentals import ANGLE_PI
from graphix.random_objects import rand_circuit
from graphix.transpiler import Circuit

if TYPE_CHECKING:
    from graphix import Pattern
    from graphix.sim.statevec import Statevec

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


def test_graph_equality(fx_rng: Generator) -> None:
    seed = int.from_bytes(fx_rng.integers(0, 256, size=16, dtype=np.uint8).tobytes())
    g = clifford_t(4, 10, 0.1, seed=seed)

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


def simulate_pattern(pattern: Pattern, rng: Generator) -> Statevec:
    pattern.remove_input_nodes()
    pattern.perform_pauli_measurements()
    pattern.minimize_space()
    return pattern.simulate_pattern(rng=rng)


def check_round_trip(pattern: Pattern, rng: Generator, full_reduce: bool) -> bool:
    opengraph = pattern.extract_opengraph()
    zx_graph = to_pyzx_graph(opengraph.to_bloch())
    if full_reduce:
        zx_graph.normalize()
        zx.simplify.full_reduce(zx_graph)
    opengraph2 = from_pyzx_graph(zx_graph)
    pattern2 = opengraph2.infer_pauli_measurements().to_pattern()
    state = simulate_pattern(pattern, rng)
    state2 = simulate_pattern(pattern2, rng)
    return state.isclose(state2)


@pytest.mark.parametrize("jumps", range(1, 11))
@pytest.mark.parametrize("full_reduce", [False, True])
def test_random_circuit(fx_bg: PCG64, jumps: int, full_reduce: bool) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 5
    depth = 5
    circuit = rand_circuit(nqubits, depth, rng, use_rzz=True)
    pattern = circuit.transpile().pattern
    assert check_round_trip(pattern, rng, full_reduce)


def test_rz(fx_rng: Generator) -> None:
    circuit = Circuit(2)
    circuit.rz(0, ANGLE_PI / 4)
    pattern = circuit.transpile().pattern
    # pyzx 0.8 does not support arithmetic expressions such as `pi / 4`.
    circ = zx.qasm(f"qreg q[2]; rz({np.pi / 4}) q[0];")  # type: ignore[attr-defined]
    g = circ.to_graph()
    og = from_pyzx_graph(g).infer_pauli_measurements()
    pattern_zx = og.to_pattern()
    state = pattern.simulate_pattern(rng=fx_rng)
    state_zx = pattern_zx.simulate_pattern(rng=fx_rng)
    assert state_zx.isclose(state)


@pytest.mark.parametrize("full_reduce", [False, True])
def test_ccx(fx_rng: Generator, full_reduce: bool) -> None:
    # Issue #235
    circuit = Circuit(3)
    circuit.ccx(0, 1, 2)
    pattern = circuit.transpile().pattern
    assert check_round_trip(pattern, fx_rng, full_reduce)
