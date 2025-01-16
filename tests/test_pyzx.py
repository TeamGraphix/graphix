from __future__ import annotations

import importlib.util  # Use fully-qualified import to avoid name conflict (util)
import random
from copy import deepcopy
from typing import TYPE_CHECKING

import pytest

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
