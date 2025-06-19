"""Flow finding algorithm.

For a given underlying graph (G, I, O, meas), this method finds a (generalized) flow [NJP 9, 250 (2007)]
in polynomincal time.
In particular, this outputs gflow with minimum depth, maximally delayed gflow.

Ref: Mhalla and Perdrix, International Colloquium on Automata,
Languages, and Programming (Springer, 2008), pp. 857-868.
Ref: Backens et al., Quantum 5, 421 (2021).

"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import typing_extensions
from fastflow import flow as flow_module
from fastflow import gflow as gflow_module
from fastflow import pflow as pflow_module
from fastflow.common import FlowResult, GFlowResult
from fastflow.common import Plane as Plane_
from fastflow.common import PPlane as PPlane_

from graphix.fundamentals import Axis, Plane
from graphix.measurements import Measurement, PauliMeasurement

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from collections.abc import Set as AbstractSet

    import networkx as nx


# MEMO: We could add layer inference

Flow = FlowResult[int]
GFlow = GFlowResult[int]
PauliFlow = GFlowResult[int]
AnyMeasurement = Measurement | PauliMeasurement


@functools.singledispatch
def _pp_convert(p: AnyMeasurement) -> PPlane_:  # noqa: ARG001
    raise TypeError


@_pp_convert.register
def _(p: Measurement) -> PPlane_:
    if p.plane == Plane.XY:
        return PPlane_.XY
    if p.plane == Plane.YZ:
        return PPlane_.YZ
    if p.plane == Plane.XZ:
        return PPlane_.XZ
    typing_extensions.assert_never(p.plane)


@_pp_convert.register
def _(p: PauliMeasurement) -> PPlane_:
    if p.axis == Axis.X:
        return PPlane_.X
    if p.axis == Axis.Y:
        return PPlane_.Y
    if p.axis == Axis.Z:
        return PPlane_.Z
    typing_extensions.assert_never(p.axis)


def _p_convert(p: Measurement) -> Plane_:
    if p.plane == Plane.XY:
        return Plane_.XY
    if p.plane == Plane.YZ:
        return Plane_.YZ
    if p.plane == Plane.XZ:
        return Plane_.XZ
    typing_extensions.assert_never(p.plane)


def _default_construct(keys: Iterable[int]) -> dict[int, Measurement]:
    # Random angle for safety
    return dict.fromkeys(
        keys,
        Measurement(Plane.XY, 0.5014943209046647),
    )


def odd_neighbor(graph: nx.Graph[int], vset: AbstractSet[int]) -> set[int]:
    """Return the odd neighbors of `vset` in `graph`."""
    odd_neighbors: set[int] = set()
    for vertex in vset:
        odd_neighbors.symmetric_difference_update(graph.neighbors(vertex))
    return odd_neighbors


def group_layers(l_k: Mapping[int, int]) -> tuple[int, dict[int, set[int]]]:
    """Group nodes by their layers.

    Parameters
    ----------
    l_k: dict
        layers obtained by flow or gflow algorithms

    Returns
    -------
    d: int
        minimum depth of graph
    layers: dict of set
        components of each layer
    """
    d = max(l_k.values())
    layers: dict[int, set[int]] = {k: set() for k in range(d + 1)}
    for i, val in l_k.items():
        layers[val].add(i)
    return d, layers


def find_flow(
    graph: nx.Graph[int],
    iset: AbstractSet[int],
    oset: AbstractSet[int],
    meas: Mapping[int, Measurement] | None = None,
) -> Flow | None:
    """Causal flow finding algorithm.

    For open graph g with input, output, and measurement planes, this returns causal flow.
    For more detail of causal flow, see Danos and Kashefi, PRA 74, 052310 (2006).

    Original algorithm by Mhalla and Perdrix,
    International Colloquium on Automata, Languages, and Programming (2008),
    pp. 857-868.

    Parameters
    ----------
    graph : `networkx.Graph`
        The underlying graph.
    iset : `collections.abc.Set`
        Input nodes.
    oset : `collections.abc.Set`
        Output nodes.

    Returns
    -------
    Flow function and layer if found, otherwise `None`.
    """
    # meas is left undocumented because it is essentially redundant.
    if meas is None:
        meas = _default_construct(graph.nodes - oset)
    if any(_p_convert(v) != Plane_.XY for v in meas.values()):
        return None
    return flow_module.find(graph, iset, oset)


def find_gflow(
    graph: nx.Graph[int],
    iset: AbstractSet[int],
    oset: AbstractSet[int],
    meas: Mapping[int, Measurement] | None = None,
) -> GFlow | None:
    """Maximally delayed gflow finding algorithm.

    For open graph g with input, output, and measurement planes, this returns maximally delayed gflow.

    gflow consist of function g(i) where i is the qubit labels,
    and strict partial ordering < or layers labels l_k where each element
    specify the order of qubits to be measured to maintain determinism in MBQC.
    In practice, we must measure qubits in order specified in array l_k (increasing order
    of l_k from 1), and for each measurements of qubit i we must perform corrections on
    qubits in g(i), depending on the measurement outcome.

    For more details of gflow, see Browne et al., NJP 9, 250 (2007).
    We use the extended gflow finding algorithm in Backens et al., Quantum 5, 421 (2021).

    Parameters
    ----------
    graph : `networkx.Graph`
        The underlying graph.
    iset : `collections.abc.Set`
        Input nodes.
    oset : `collections.abc.Set`
        Output nodes.
    meas : `collections.abc.Mapping`, optional
        Measurement specs for each qubit, by default all XY.

    Returns
    -------
    Gflow function and layer if found, otherwise `None`.
    """
    if meas is None:
        meas = _default_construct(graph.nodes - oset)
    return gflow_module.find(graph, iset, oset, {k: _p_convert(v) for k, v in meas.items()})


def find_pauliflow(
    graph: nx.Graph[int],
    iset: AbstractSet[int],
    oset: AbstractSet[int],
    meas: Mapping[int, AnyMeasurement] | None = None,
) -> PauliFlow | None:
    """Maximally delayed Pauli flow finding algorithm.

    For open graph g with input, output, measurement planes and measurement angles, this returns maximally delayed Pauli flow.

    Pauli flow consist of function p(i) where i is the qubit labels,
    and strict partial ordering < or layers labels l_k where each element
    specify the order of qubits to be measured to maintain determinism in MBQC.
    In practice, we must measure qubits in order specified in array l_k (increasing order
    of l_k from 1), and for each measurements of qubit i we must perform corrections on
    qubits in p(i), depending on the measurement outcome.

    For more details of Pauli flow and the finding algorithm used in this method,
    see Simmons et al., EPTCS 343, 2021, pp. 50-101 (arXiv:2109.05654).

    Parameters
    ----------
    graph : `networkx.Graph`
        The underlying graph.
    iset : `collections.abc.Set`
        Input nodes.
    oset : `collections.abc.Set`
        Output nodes.
    meas : `collections.abc.Mapping`, optional
        Measurement specs for each qubit, by default all XY.

    Returns
    -------
    Pauli flow function and layer if found, otherwise `None`.
    """
    if meas is None:
        meas = _default_construct(graph.nodes - oset)
    return pflow_module.find(graph, iset, oset, {k: _pp_convert(v) for k, v in meas.items()})


def verify_flow(
    flow: Flow,
    graph: nx.Graph[int],
    iset: AbstractSet[int],
    oset: AbstractSet[int],
    *,
    allow_raise: bool = False,
) -> bool:
    """Check whether the flow is valid.

    Parameters
    ----------
    flow
        The flow to verify.
    graph : `networkx.Graph`
        The underlying graph.
    iset : `collections.abc.Set`
        Input nodes.
    oset : `collections.abc.Set`
        Output nodes.
    allow_raise : bool, optional
        Whether to allow raising an exception on failure, by default `False`.

    Returns
    -------
    bool
        Whether the flow is valid.
    """
    try:
        flow_module.verify(flow, graph, iset, oset)
    except ValueError as e:
        if not allow_raise:
            return False
        msg = "Flow verification failed."
        raise ValueError(msg) from e
    return True


def verify_gflow(
    gflow: GFlow,
    graph: nx.Graph[int],
    iset: AbstractSet[int],
    oset: AbstractSet[int],
    meas: Mapping[int, Measurement] | None = None,
    *,
    allow_raise: bool = False,
) -> bool:
    """Check whether the gflow is valid.

    Parameters
    ----------
    gflow
        The gflow to verify.
    graph : `networkx.Graph`
        The underlying graph.
    iset : `collections.abc.Set`
        Input nodes.
    oset : `collections.abc.Set`
        Output nodes.
    meas : `collections.abc.Mapping`, optional
        Measurement specs for each qubit, by default all XY.

    Returns
    -------
    bool
        Whether the gflow is valid.
    """
    if meas is None:
        meas = _default_construct(graph.nodes - oset)
    try:
        gflow_module.verify(gflow, graph, iset, oset, {k: _p_convert(v) for k, v in meas.items()})
    except ValueError as e:
        if not allow_raise:
            return False
        msg = "GFlow verification failed."
        raise ValueError(msg) from e
    return True


def verify_pauliflow(
    pflow: PauliFlow,
    graph: nx.Graph[int],
    iset: AbstractSet[int],
    oset: AbstractSet[int],
    meas: Mapping[int, AnyMeasurement] | None = None,
    *,
    allow_raise: bool = False,
) -> bool:
    """Check whether the Pauliflow is valid.

    Parameters
    ----------
    pflow
        The Pauliflow to verify.
    graph : `networkx.Graph`
        The underlying graph.
    iset : `collections.abc.Set`
        Input nodes.
    oset : `collections.abc.Set`
        Output nodes.
    meas : `collections.abc.Mapping`, optional
        Measurement specs for each qubit, by default all XY.
    allow_raise : bool, optional
        Whether to allow raising an exception on failure, by default `False`.

    Returns
    -------
    bool
        Whether the Pauliflow is valid.
    """
    if meas is None:
        meas = _default_construct(graph.nodes - oset)
    try:
        pflow_module.verify(pflow, graph, iset, oset, {k: _pp_convert(v) for k, v in meas.items()})
    except ValueError as e:
        if not allow_raise:
            return False
        msg = "PauliFlow verification failed."
        raise ValueError(msg) from e
    return True
