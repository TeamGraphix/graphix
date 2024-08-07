from __future__ import annotations

import os
import sys

import networkx as nx
import pytest

from graphix.opengraph import Measurement, OpenGraph
from graphix.pauli import Plane

try:
    import pyzx as zx

    from graphix.pyzx import from_pyzx_graph, to_pyzx_graph
except ModuleNotFoundError:
    pass


@pytest.mark.skipif(sys.modules.get("pyzx") is None, reason="pyzx not installed")
def test_graph_no_output_measurements() -> None:
    g = nx.Graph([(0, 1)])
    meas = {0: Measurement(0, Plane.XY), 1: Measurement(0, Plane.XY)}
    inputs = [0]
    outputs = [1]

    # Output node can not be measurement
    with pytest.raises(ValueError):
        OpenGraph(g, meas, inputs, outputs)


@pytest.mark.skipif(sys.modules.get("pyzx") is None, reason="pyzx not installed")
def test_graph_equality() -> None:
    file = "./tests/circuits/adder_n4.qasm"
    circ = zx.Circuit.load(file)

    g = circ.to_graph()
    og1 = from_pyzx_graph(g)

    g_copy = circ.to_graph()
    og2 = from_pyzx_graph(g_copy)

    assert og1 == og2


# Converts a graph to and from an Open graph and then checks the resulting
# pyzx graph is equal to the original.
@pytest.mark.skipif(sys.modules.get("pyzx") is None, reason="pyzx not installed")
def assert_reconstructed_pyzx_graph_equal(circ) -> None:
    g = circ.to_graph()
    zx.simplify.to_graph_like(g)

    g_copy = circ.to_graph()
    og = from_pyzx_graph(g_copy)
    reconstructed_pyzx_graph = to_pyzx_graph(og)

    # The "tensorfy" function break if the rows aren't set for some reason
    for v in reconstructed_pyzx_graph.vertices():
        reconstructed_pyzx_graph.set_row(v, 2)

    ten = zx.tensorfy(g)
    ten_graph = zx.tensorfy(reconstructed_pyzx_graph)
    assert zx.compare_tensors(ten, ten_graph)


@pytest.fixture
def all_small_circuits():
    direc = "./tests/circuits/"
    directory = os.fsencode(direc)

    circuits = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if not filename.endswith(".qasm"):
            raise Exception(f"only '.qasm' files allowed: not {filename}")

        circ = zx.Circuit.load(direc + filename)
        circuits.append(circ)

    return circuits


# Tests that compiling from a pyzx graph to an OpenGraph returns the same
# graph. Only works with small circuits up to 4 qubits since PyZX's `tensorfy`
# function seems to consume huge amount of memory for larger qubit
@pytest.mark.skipif(sys.modules.get("pyzx") is None, reason="pyzx not installed")
def test_all_small_circuits(all_small_circuits) -> None:
    for circ in all_small_circuits:
        assert_reconstructed_pyzx_graph_equal(circ)