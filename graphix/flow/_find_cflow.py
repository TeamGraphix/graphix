"""Causal flow finding algorithm.

This module implements Algorithm 1 from Ref. [1]. For a given labelled open graph (G, I, O, meas_plane), this algorithm finds a causal flow [2] in polynomial time with the number of nodes, :math:`O(N^2)`.

References
----------
[1] Mhalla and Perdrix, (2008), Finding Optimal Flows Efficiently, doi.org/10.1007/978-3-540-70575-8_70
[2] Browne et al., 2007 New J. Phys. 9 250 (arXiv:quant-ph/0702212)
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

from graphix.fundamentals import Axis, Plane

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet

    import networkx as nx

    from graphix.flow.core import CausalFlow, ExtendedCausalFlow
    from graphix.fundamentals import AbstractMeasurement
    from graphix.opengraph import OpenGraph, _AM_co, _PM_co


class ECFlowLabel(enum.Flag):
    """Labels that a measurement can take in an extended causal flow."""

    NONE = 0
    XY = enum.auto()
    YZ = enum.auto()


def ecflow_labels(measurement: AbstractMeasurement) -> ECFlowLabel:
    """Return the labels that ``measurement`` can take in an extended causal flow.

    The labels follow the convention of Pauli flow (see
    :meth:`~graphix.fundamentals.AbstractMeasurement.to_plane_or_axis`): a planar
    measurement (e.g., a :class:`~graphix.measurements.BlochMeasurement`, even with a Pauli
    angle) is labelled by its plane, and a Pauli measurement (e.g., an inferred
    :class:`~graphix.measurements.PauliMeasurement`) can be labelled with any of the planes
    :math:`XY` and :math:`YZ` that contain it. A Pauli :math:`X` (resp. :math:`Z`)
    measurement is labelled :math:`XY` (resp. :math:`YZ`), and a Pauli :math:`Y`
    measurement can be labelled either: the extended causal flow then determines the label,
    :math:`YZ` if and only if the node is its own corrector.

    Parameters
    ----------
    measurement : AbstractMeasurement
        A measurement, a plane or an axis.

    Returns
    -------
    ECFlowLabel
        The allowed labels (``ECFlowLabel.NONE`` for :math:`XZ`-measurements).
    """
    match measurement.to_plane_or_axis():
        case Plane.XY | Axis.X:
            return ECFlowLabel.XY
        case Plane.YZ | Axis.Z:
            return ECFlowLabel.YZ
        case Axis.Y:
            return ECFlowLabel.XY | ECFlowLabel.YZ
        case _:
            return ECFlowLabel.NONE


def ecflow_label_sets(measurements: Mapping[int, AbstractMeasurement]) -> tuple[set[int], set[int]] | None:
    """Return the nodes that can be labelled :math:`XY` and those that can be labelled :math:`YZ`.

    Parameters
    ----------
    measurements : Mapping[int, AbstractMeasurement]
        The measurements of the measured nodes.

    Returns
    -------
    tuple[set[int], set[int]] | None
        The pair ``(xy_nodes, yz_nodes)`` (Pauli :math:`Y`-measured nodes belong to both),
        or ``None`` if some node cannot be labelled (:math:`XZ`-measurement).
    """
    xy_nodes: set[int] = set()
    yz_nodes: set[int] = set()
    for node, measurement in measurements.items():
        labels = ecflow_labels(measurement)
        if not labels:
            return None
        if ECFlowLabel.XY in labels:
            xy_nodes.add(node)
        if ECFlowLabel.YZ in labels:
            yz_nodes.add(node)
    return xy_nodes, yz_nodes


def find_ecflow_layers(
    graph: nx.Graph[int],
    input_nodes: AbstractSet[int],
    output_nodes: AbstractSet[int],
    xy_nodes: AbstractSet[int],
    yz_nodes: AbstractSet[int] = frozenset(),
) -> tuple[dict[int, frozenset[int]], list[frozenset[int]]] | None:
    r"""Compute the correction function and layers of a maximally delayed (extended) causal flow.

    The graph is traversed backward from the outputs:

    - a node ``q`` of ``xy_nodes`` is corrected by a node ``p`` which is neither an input nor
      self-corrected, and whose only uncorrected neighbour is ``q``: this is condition (C.XY)
      of Definition 10 in Ref. [1] (reported as :attr:`~graphix.flow.exceptions.FlowPropositionErrorReason.C1`);
    - a non-input node ``q`` of ``yz_nodes`` is self-corrected as soon as all its neighbours are
      corrected: this is condition (C.YZ) of Definition 10 in Ref. [1] (reported as
      :attr:`~graphix.flow.exceptions.FlowPropositionErrorReason.CYZ`), and
      :math:`R_f` requires ``q`` to precede all its neighbours.

    A node may belong to both ``xy_nodes`` and ``yz_nodes`` (e.g., a Pauli :math:`Y`
    measurement): it is then labelled :math:`XY` if possible, which is never worse since an
    :math:`XY`-node can correct other nodes. Self-corrected nodes (``q`` in ``cf[q]``) are the
    nodes labelled :math:`YZ`.

    Parameters
    ----------
    graph : nx.Graph[int]
        Underlying graph.
    input_nodes : AbstractSet[int]
        Input nodes.
    output_nodes : AbstractSet[int]
        Output nodes.
    xy_nodes : AbstractSet[int]
        Measured nodes that can be labelled :math:`XY`.
    yz_nodes : AbstractSet[int]
        Measured nodes that can be labelled :math:`YZ`.

    Returns
    -------
    tuple[dict[int, frozenset[int]], list[frozenset[int]]] | None
        The correction function and the partial-order layers (outputs first), or ``None`` if no
        (extended) causal flow exists.

    References
    ----------
    [1] Backens and Perdrix, (2026), Completeness for flow-preserving rewrite rules, arXiv:2608.13035
    """
    graph_nodes = set(graph.nodes)
    corrected_nodes = set(output_nodes)
    corrector_candidates = corrected_nodes - input_nodes
    yz_candidates = set(yz_nodes) - input_nodes

    cf: dict[int, frozenset[int]] = {}
    # Output nodes are always in layer 0.
    # If the open graph has causal flow, it must have outputs (unless it is an empty open graph).
    layers: list[frozenset[int]] = [frozenset(output_nodes)] if output_nodes else []

    while corrected_nodes != graph_nodes:
        corrected_nodes_new: set[int] = set()
        corrector_nodes_new: set[int] = set()

        for p in corrector_candidates:
            match [*(set(graph.neighbors(p)) - corrected_nodes)]:
                case [q] if q in xy_nodes:
                    cf[q] = frozenset({p})
                    corrected_nodes_new.add(q)
                    corrector_nodes_new.add(p)

        for q in yz_candidates - corrected_nodes_new:
            if set(graph.neighbors(q)) <= corrected_nodes:
                cf[q] = frozenset({q})
                corrected_nodes_new.add(q)

        if not corrected_nodes_new:
            return None

        layers.append(frozenset(corrected_nodes_new))
        corrected_nodes |= corrected_nodes_new
        yz_candidates -= corrected_nodes_new
        # Self-corrected nodes can never correct another node: the corrected node would be both before and after them.
        corrector_candidates = (corrector_candidates - corrector_nodes_new) | {
            q for q in corrected_nodes_new if q not in input_nodes and q not in cf[q]
        }
    return cf, layers


def find_cflow(og: OpenGraph[_PM_co]) -> CausalFlow[_PM_co] | None:
    """Return the causal flow of the input open graph if it exists.

    Parameters
    ----------
    og : OpenGraph[_PM_co]
        Open graph whose causal flow is calculated.

    Returns
    -------
    CausalFlow[_PM_co] | None
        A causal flow object if the open graph has causal flow, `None` otherwise.

    Notes
    -----
    - See Definition 2, Theorem 1 and Algorithm 1 in Ref. [1].
    - The open graph instance must be of parametric type `Measurement` or `Plane` since the causal flow is only defined on open graphs with :math:`XY` measurements.

    References
    ----------
    [1] Mhalla and Perdrix, (2008), Finding Optimal Flows Efficiently, doi.org/10.1007/978-3-540-70575-8_70
    """
    from graphix.flow.core import CausalFlow  # noqa: PLC0415 (circular import)

    for measurement in og.measurements.values():
        if measurement.to_plane() in {Plane.XZ, Plane.YZ}:
            return None

    result = find_ecflow_layers(og.graph, set(og.input_nodes), set(og.output_nodes), og.measurements.keys())
    if result is None:
        return None
    cf, layers = result
    return CausalFlow(og, cf, tuple(layers))


def find_ecflow(og: OpenGraph[_AM_co]) -> ExtendedCausalFlow[_AM_co] | None:
    """Return the extended causal flow of the input open graph if it exists.

    Parameters
    ----------
    og : OpenGraph[_AM_co]
        Open graph whose extended causal flow is calculated.

    Returns
    -------
    ExtendedCausalFlow[_AM_co] | None
        An extended causal flow object if the open graph has extended causal flow, `None` otherwise.

    Notes
    -----
    The algorithm is derived from Algorithm 1 in Ref. [1], see :func:`find_ecflow_layers`.
    Pauli measurements (i.e., measurements interpreted as axes, such as inferred
    :class:`~graphix.measurements.PauliMeasurement`) can be labelled with any plane
    containing them, see :func:`ecflow_labels`.

    References
    ----------
    [1] Mhalla and Perdrix, (2008), Finding Optimal Flows Efficiently, doi.org/10.1007/978-3-540-70575-8_70
    [2] Backens and Perdrix, (2026), Completeness for flow-preserving rewrite rules, arXiv:2608.13035
    """
    from graphix.flow.core import ExtendedCausalFlow  # noqa: PLC0415 (circular import)

    label_sets = ecflow_label_sets(og.measurements)
    if label_sets is None:
        return None
    result = find_ecflow_layers(og.graph, set(og.input_nodes), set(og.output_nodes), *label_sets)
    if result is None:
        return None
    cf, layers = result
    return ExtendedCausalFlow(og, cf, tuple(layers))
