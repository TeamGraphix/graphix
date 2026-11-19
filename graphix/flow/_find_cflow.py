"""Causal flow and extended causal flow finding algorithms.

This module implements Algorithm 1 from Ref. [1] and an adaptation of it to extended causal flows [3]. For a given labelled open graph (G, I, O, meas_plane), these algorithms find a causal flow [2] or an extended causal flow [3] in polynomial time with the number of nodes, :math:`O(N^2)`.

References
----------
[1] Mhalla and Perdrix, (2008), Finding Optimal Flows Efficiently, doi.org/10.1007/978-3-540-70575-8_70
[2] Browne et al., 2007 New J. Phys. 9 250 (arXiv:quant-ph/0702212)
[3] Backens and Perdrix, (2026), Completeness for flow-preserving rewrite rules, arXiv:2608.13035
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from graphix.flow.core import CausalFlow, ExtendedCausalFlow
from graphix.fundamentals import Plane

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet
    from typing import TypeAlias, TypeVar

    from graphix.fundamentals import AbstractPlanarMeasurement
    from graphix.opengraph import OpenGraph

    FlowArguments: TypeAlias = tuple[dict[int, frozenset[int]], tuple[frozenset[int], ...]]
    _PM_co = TypeVar("_PM_co", bound=AbstractPlanarMeasurement, covariant=True)


def find_cflow(og: OpenGraph[_PM_co]) -> CausalFlow[_PM_co] | None:
    """Return the causal flow of the input open graph if it exists.

    Parameters
    ----------
    og : OpenGraph[_PM_co]
        Open graph whose causal flow is calculated.

    Returns
    -------
    CausalFlow[_PM_co] | None
        A causal flow object if the open graph has causal flow, ``None`` otherwise.

    Notes
    -----
    - See Definition 2, Theorem 1 and Algorithm 1 in Ref. [1].
    - Causal flow is only defined on open graphs with :math:`XY` measurements: ``None`` is returned if some measurement is in the :math:`XZ` or :math:`YZ` plane. Measurements are labelled by :meth:`AbstractPlanarMeasurement.to_plane`.

    References
    ----------
    [1] Mhalla and Perdrix, (2008), Finding Optimal Flows Efficiently, doi.org/10.1007/978-3-540-70575-8_70
    """
    for measurement in og.measurements.values():
        if measurement.to_plane() in {Plane.XZ, Plane.YZ}:
            return None

    flow_args = _find_flow(og)
    if flow_args is None:
        return None
    return CausalFlow(og, *flow_args)


def find_ecflow(og: OpenGraph[_PM_co]) -> ExtendedCausalFlow[_PM_co] | None:
    """Return the extended causal flow of the input open graph if it exists.

    Parameters
    ----------
    og : OpenGraph[_PM_co]
        Open graph whose extended causal flow is calculated.

    Returns
    -------
    ExtendedCausalFlow[_PM_co] | None
        An extended causal flow object if the open graph has extended causal flow, ``None`` otherwise.

    Notes
    -----
    - See Definition 10 in Ref. [1]. The algorithm is an adaptation of Algorithm 1 in Ref. [2] where, at each step, a :math:`YZ`-measured node is self-corrected and added to the current layer as soon as all its neighbors have been corrected. :math:`YZ`-measured nodes never correct other nodes.
    - Extended causal flow is only defined on open graphs with :math:`XY` and :math:`YZ` measurements: ``None`` is returned if some measurement is in the :math:`XZ` plane. Measurements are labelled by :meth:`AbstractPlanarMeasurement.to_plane`.

    References
    ----------
    [1] Backens and Perdrix, (2026), Completeness for flow-preserving rewrite rules, arXiv:2608.13035
    [2] Mhalla and Perdrix, (2008), Finding Optimal Flows Efficiently, doi.org/10.1007/978-3-540-70575-8_70
    """
    yz_candidates: set[int] = set()
    for node, measurement in og.measurements.items():
        plane = measurement.to_plane()
        if plane == Plane.XZ:
            return None
        if plane == Plane.YZ:
            yz_candidates.add(node)

    flow_args = _find_flow(og, yz_candidates)
    if flow_args is None:
        return None
    return ExtendedCausalFlow(og, *flow_args)


def _find_flow(og: OpenGraph[_PM_co], yz_candidates: AbstractSet[int] | None = None) -> FlowArguments | None:
    """Compute the correction function and the partial order layers of a causal or an extended causal flow.

    This is the routine shared by :func:`find_cflow` and :func:`find_ecflow`. Nodes not in ``yz_candidates`` are treated as :math:`XY`-measured nodes and are corrected by a single neighbor, as in a causal flow. Nodes in ``yz_candidates`` are treated as :math:`YZ`-measured nodes: they are corrected by themselves and never correct other nodes.

    Parameters
    ----------
    og : OpenGraph[_PM_co]
        Open graph whose flow is calculated. The caller is responsible for discarding open graphs with measurements in the :math:`XZ` plane (and in the :math:`YZ` plane when looking for a causal flow).
    yz_candidates : AbstractSet[int] | None, optional
        Non-output nodes measured in the :math:`YZ` plane. The set is not modified. If ``None`` (default), it is considered to be empty.

    Returns
    -------
    FlowArguments | None
        A tuple ``(correction_function, partial_order_layers)`` used to instantiate a :class:`CausalFlow` or an :class:`ExtendedCausalFlow` if the flow exists, ``None`` otherwise.

    Notes
    -----
    See Definition 2, Theorem 1 and Algorithm 1 in Ref. [1] for causal flow, and Definition 10 in Ref. [2] for extended causal flow.

    References
    ----------
    [1] Mhalla and Perdrix, (2008), Finding Optimal Flows Efficiently, doi.org/10.1007/978-3-540-70575-8_70
    [2] Backens and Perdrix, (2026), Completeness for flow-preserving rewrite rules, arXiv:2608.13035
    """
    yz_candidates = set(yz_candidates) if yz_candidates is not None else set()

    graph_nodes = set(og.graph.nodes)
    corrected_nodes = set(og.output_nodes)
    iset = set(og.input_nodes)
    corrector_candidates = corrected_nodes - iset

    cf: dict[int, frozenset[int]] = {}
    # Output nodes are always in layer 0.
    # If the open graph has causal flow, it must have outputs (unless it is an empty open graph).
    layers: list[frozenset[int]] = [frozenset(og.output_nodes)] if og.output_nodes else []

    while corrected_nodes != graph_nodes:
        corrected_nodes_new: set[int] = set()
        corrector_nodes_new: set[int] = set()

        for p in corrector_candidates:
            non_corrected_neighbors = og.neighbors({p}) - corrected_nodes
            if len(non_corrected_neighbors) == 1:
                (q,) = non_corrected_neighbors
                if og.measurements[q].to_plane() == Plane.XY:
                    cf[q] = frozenset({p})
                    corrected_nodes_new.add(q)
                    corrector_nodes_new.add(p)

        for q in yz_candidates - corrected_nodes_new:
            if og.neighbors({q}) <= corrected_nodes:
                cf[q] = frozenset({q})
                corrected_nodes_new.add(q)

        if len(corrected_nodes_new) == 0:
            return None

        layers.append(frozenset(corrected_nodes_new))
        corrected_nodes |= corrected_nodes_new
        yz_candidates -= corrected_nodes_new
        corrector_candidates = (corrector_candidates - corrector_nodes_new) | {
            q for q in corrected_nodes_new if q not in iset and q not in cf[q]
        }

    return cf, tuple(layers)
