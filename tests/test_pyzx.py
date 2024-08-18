from __future__ import annotations

import random
import sys
from copy import deepcopy

import pytest
from pyzx.generate import cliffordT

try:
    import pyzx as zx

    from graphix.pyzx import from_pyzx_graph, to_pyzx_graph
except ModuleNotFoundError:
    pass

SEED = 123

@pytest.mark.skipif(sys.modules.get("pyzx") is None, reason="pyzx not installed")
def test_graph_equality() -> None:
    random.seed(SEED)
    g = cliffordT(4, 10, 0.1)

    og1 = from_pyzx_graph(g)

    g_copy = deepcopy(g)
    og2 = from_pyzx_graph(g_copy)

    assert og1 == og2


# Converts a graph to and from an Open graph and then checks the resulting
# pyzx graph is equal to the original.
@pytest.mark.skipif(sys.modules.get("pyzx") is None, reason="pyzx not installed")
def assert_reconstructed_pyzx_graph_equal(g) -> None:
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
@pytest.mark.skipif(sys.modules.get("pyzx") is None, reason="pyzx not installed")
def test_random_cliffordT() -> None:
    for i in range(15):
        g = cliffordT(4, 10, 0.1)
        assert_reconstructed_pyzx_graph_equal(g)
