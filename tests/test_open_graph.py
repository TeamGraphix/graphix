import os

import numpy as np
import pyzx as zx

from graphix.open_graph import OpenGraph


def test_graph_equality():
    file = "./tests/circuits/adder_n4.qasm"
    circ = zx.Circuit.load(file)

    g = circ.to_graph()
    og1 = OpenGraph.from_pyzx_graph(g)

    g_copy = circ.to_graph()
    og2 = OpenGraph.from_pyzx_graph(g_copy)

    assert og1 == og2


# Converts a graph to and from an Open graph and then checks the resulting
# pyzx graph is equal to the original.
def assert_reconstructed_pyzx_graph_equal(circ: zx.Circuit):
    g = circ.to_graph()
    zx.simplify.to_graph_like(g)
    zx.simplify.full_reduce(g)

    g_copy = circ.to_graph()
    og = OpenGraph.from_pyzx_graph(g_copy)
    reconstructed_pyzx_graph = og.to_pyzx_graph()

    # The "tensorfy" function break if the rows aren't set for some reason
    for v in reconstructed_pyzx_graph.vertices():
        reconstructed_pyzx_graph.set_row(v, 2)

    ten = zx.tensorfy(g).flatten()
    ten_graph = zx.tensorfy(reconstructed_pyzx_graph).flatten()

    # Here we check their tensor representations instead of composing g with
    # the adjoint of reconstructed_pyzx_graph and checking it reduces to the
    # identity since there seems to be a bug where equal graphs don't produce
    # the identity
    i = np.argmax(ten)
    assert np.allclose(ten / ten[i], ten_graph / ten_graph[i])


# Tests that compiling from a pyzx graph to an OpenGraph returns the same
# graph. Only works with small circuits up to 4 qubits since PyZX's `tensorfy`
# function seems to consume huge amount of memory for larger qubit
def test_all_small_circuits():
    direc = "./tests/circuits/"
    directory = os.fsencode(direc)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if not filename.endswith(".qasm"):
            raise Exception(f"only '.qasm' files allowed: not {filename}")

        circ = zx.Circuit.load(direc + filename)
        assert_reconstructed_pyzx_graph_equal(circ)
