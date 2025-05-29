"""Functionality for converting between OpenGraphs and PyZX.

These functions are held in their own file rather than including them in the
OpenGraph class because we want PyZX to be an optional dependency.
"""

from __future__ import annotations

import warnings
from fractions import Fraction
from typing import TYPE_CHECKING, SupportsFloat

import networkx as nx
import pyzx as zx
from pyzx.graph import Graph
from pyzx.utils import EdgeType, FractionLike, VertexType

from graphix.fundamentals import Plane
from graphix.measurements import Measurement
from graphix.opengraph import OpenGraph

if TYPE_CHECKING:
    from pyzx.graph.base import BaseGraph


def to_pyzx_graph(og: OpenGraph) -> BaseGraph[int, tuple[int, int]]:
    """Return a PyZX graph corresponding to the the open graph.

    Example
    -------
    >>> import networkx as nx
    >>> from graphix.pyzx import to_pyzx_graph
    >>> g = nx.Graph([(0, 1), (1, 2)])
    >>> inputs = [0]
    >>> outputs = [2]
    >>> measurements = {0: Measurement(0, Plane.XY), 1: Measurement(1, Plane.YZ)}
    >>> og = OpenGraph(g, measurements, inputs, outputs)
    >>> reconstructed_pyzx_graph = to_pyzx_graph(og)
    """
    # check pyzx availability and version
    try:
        import pyzx as zx
    except ModuleNotFoundError as e:
        msg = "Cannot find pyzx (optional dependency)."
        raise RuntimeError(msg) from e
    if zx.__version__ != "0.8.0":
        warnings.warn(
            "`to_pyzx_graph` is guaranteed to work only with pyzx==0.8.0 due to possible breaking changes in `pyzx`.",
            stacklevel=1,
        )
    g = Graph()

    # Add vertices into the graph and set their type
    def add_vertices(n: int, ty: VertexType.Type) -> list[VertexType]:
        verts = g.add_vertices(n)
        for vert in verts:
            g.set_type(vert, ty)

        return verts

    # Add input boundary nodes
    in_verts = add_vertices(len(og.inputs), VertexType.BOUNDARY)
    g.set_inputs(tuple(in_verts))

    # Add nodes for internal Z spiders - not including the phase gadgets
    body_verts = add_vertices(len(og.inside), VertexType.Z)

    # Add nodes for the phase gadgets. In OpenGraph we don't store the
    # effect as a seperate node, it is instead just stored in the
    # "measurement" attribute of the node it measures.
    x_meas = [i for i, m in og.measurements.items() if m.plane == Plane.YZ]
    x_meas_verts = add_vertices(len(x_meas), VertexType.Z)

    out_verts = add_vertices(len(og.outputs), VertexType.BOUNDARY)
    g.set_outputs(tuple(out_verts))

    # Maps a node's ID in the Open Graph to it's corresponding node ID in
    # the PyZX graph and vice versa.
    map_to_og = dict(zip(body_verts, og.inside.nodes()))
    map_to_pyzx = {v: i for i, v in map_to_og.items()}

    # Open Graph's don't have boundary nodes, so we need to connect the
    # input and output Z spiders to their corresponding boundary nodes in
    # pyzx.
    for pyzx_index, og_index in zip(in_verts, og.inputs):
        g.add_edge((pyzx_index, map_to_pyzx[og_index]))
    for pyzx_index, og_index in zip(out_verts, og.outputs):
        g.add_edge((pyzx_index, map_to_pyzx[og_index]))

    og_edges = og.inside.edges()
    pyzx_edges = ((map_to_pyzx[a], map_to_pyzx[b]) for a, b in og_edges)
    g.add_edges(pyzx_edges, EdgeType.HADAMARD)

    # Add the edges between the Z spiders in the graph body
    for og_index, meas in og.measurements.items():
        # If it's an X measured node, then we handle it in the next loop
        if meas.plane == Plane.XY:
            g.set_phase(map_to_pyzx[og_index], Fraction(meas.angle))

    # Connect the X measured vertices
    for og_index, pyzx_index in zip(x_meas, x_meas_verts):
        g.add_edge((map_to_pyzx[og_index], pyzx_index), EdgeType.HADAMARD)
        g.set_phase(pyzx_index, Fraction(og.measurements[og_index].angle))

    return g


def _checked_float(x: FractionLike) -> float:
    if not isinstance(x, SupportsFloat):
        # Possibly a Poly object
        raise TypeError(f"Cannot convert {x} to a float.")
    return float(x)


def from_pyzx_graph(g: BaseGraph[int, tuple[int, int]]) -> OpenGraph:
    """Construct an Optyx Open Graph from a PyZX graph.

    This method may add additional nodes to the graph so that it adheres
    with the definition of an OpenGraph. For instance, if the final node on
    a qubit is measured, it will add two nodes behind it so that no output
    nodes are measured to satisfy the requirements of an open graph.
        .. warning::
            works with `pyzx==0.8.0` (see `requirements-dev.txt`). Other versions may not be compatible due to breaking changes in `pyzx`
    Example
    -------
    >>> import pyzx as zx
    >>> from graphix.pyzx import from_pyzx_graph
    >>> circ = zx.qasm("qreg q[2]; h q[1]; cx q[0], q[1]; h q[1];")
    >>> g = circ.to_graph()
    >>> og = from_pyzx_graph(g)
    """
    zx.simplify.to_graph_like(g)

    measurements = {}
    inputs = list(g.inputs())
    outputs = list(g.outputs())

    g_nx = nx.Graph(g.edges())

    # We need to do this since the full reduce simplification can
    # leave either hadamard or plain wires on the inputs and outputs
    for inp in g.inputs():
        first_nbr = next(iter(g.neighbors(inp)))
        et = g.edge_type((first_nbr, inp))

        if et == EdgeType.SIMPLE:
            g_nx.remove_node(inp)
            inputs = [i if i != inp else first_nbr for i in inputs]

    for out in g.outputs():
        first_nbr = next(iter(g.neighbors(out)))
        et = g.edge_type((first_nbr, out))

        if et == EdgeType.SIMPLE:
            g_nx.remove_node(out)
            outputs = [o if o != out else first_nbr for o in outputs]

    # Turn all phase gadgets into measurements
    # Since we did a full reduce, any node that isn't an input or output
    # node and has only one neighbour is definitely a phase gadget.
    nodes = list(g_nx.nodes())
    for v in nodes:
        if v in inputs or v in outputs:
            continue

        nbrs = list(g.neighbors(v))
        if len(nbrs) == 1:
            measurements[nbrs[0]] = Measurement(_checked_float(g.phase(v)), Plane.YZ)
            g_nx.remove_node(v)

    next_id = max(g_nx.nodes) + 1

    # Since outputs can't be measured, we need to add an extra two nodes
    # in to counter it
    for out in outputs:
        if g.phase(out) == 0:
            continue

        g_nx.add_edges_from([(out, next_id), (next_id, next_id + 1)])
        measurements[next_id] = Measurement(0, Plane.XY)

        outputs = [o if o != out else next_id + 1 for o in outputs]
        next_id += 2

    # Add the phase to all XY measured nodes
    for v in g_nx.nodes:
        if v in outputs or v in measurements:
            continue

        # g.phase() may be a fractions.Fraction object, but Measurement
        # expects a float
        measurements[v] = Measurement(_checked_float(g.phase(v)), Plane.XY)

    return OpenGraph(g_nx, measurements, inputs, outputs)
