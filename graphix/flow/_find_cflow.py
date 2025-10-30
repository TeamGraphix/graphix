"""Causal flow finding algorithm.

This module implements Algorithm 1 from Ref. [1]. For a given labelled open graph (G, I, O, meas_plane), this algorithm finds a causal flow [2] in polynomial time with the number of nodes, :math:`O(N^2)`.

References
----------
[1] Mhalla and Perdrix, (2008), Finding Optimal Flows Efficiently, doi.org/10.1007/978-3-540-70575-8_70
[2] Browne et al., 2007 New J. Phys. 9 250 (arXiv:quant-ph/0702212)
"""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

from graphix.flow.core import CausalFlow
from graphix.fundamentals import Plane

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet

    from graphix.opengraph_ import OpenGraph, _PM_co


def find_cflow(og: OpenGraph[_PM_co]) -> CausalFlow[_PM_co] | None:
    """Return the causal flow of the input open graph if it exists.

    Parameters
    ----------
    og : OpenGraph[_PM_co]
        Open graph whose causal flow is calculated.

    Returns
    -------
    CausalFlow | None
        A causal flow object if the open graph has causal flow, `None` otherwise.

    Notes
    -----
    - See Definition 2, Theorem 1 and Algorithm 1 in Ref. [1].
    - The open graph instance must be of parametric type `Measurement` or `Plane` since the causal flow is only defined on open graphs with :math:`XY` measurements.

    References
    ----------
    [1] Mhalla and Perdrix, (2008), Finding Optimal Flows Efficiently, doi.org/10.1007/978-3-540-70575-8_70
    """
    for measurement in og.measurements.values():
        if measurement.to_plane() in {Plane.XZ, Plane.YZ}:
            return None

    corrected_nodes = set(og.output_nodes)
    corrector_candidates = corrected_nodes - set(og.input_nodes)
    non_input_nodes = og.graph.nodes - set(og.input_nodes)

    cf: dict[int, set[int]] = {}
    # Output nodes are always in layer 0. If the open graph has flow, it must have outputs, so we never end up with an empty set at `layers[0]`.
    layers: list[set[int]] = [
        copy(corrected_nodes)
    ]  # A copy is necessary because `corrected_nodes` is mutable and changes during the algorithm.

    return _flow_aux(og, non_input_nodes, corrected_nodes, corrector_candidates, cf, layers)


def _flow_aux(
    og: OpenGraph[_PM_co],
    non_input_nodes: AbstractSet[int],
    corrected_nodes: AbstractSet[int],
    corrector_candidates: AbstractSet[int],
    cf: dict[int, set[int]],
    layers: list[set[int]],
) -> CausalFlow[_PM_co] | None:
    """Find one layer of the causal flow.

    Parameters
    ----------
    og : OpenGraph[Plane]
        Open graph whose causal flow is calculated.
    non_input_nodes : AbstractSet[int]
        Non-input nodes of the input open graph. This parameter remains constant throughout the execution of the algorithm and can be derived from `og` at any time. It is passed as an argument to avoid unnecessary recalculations.
    corrected_nodes : AbstractSet[int]
        Nodes which have already been corrected.
    corrector_candidates : AbstractSet[int]
        Nodes which could correct a node at the time of calling the function. This set can never contain input nodes, uncorrected nodes or nodes which already correct another node.
    cf : dict[int, set[int]]
        Causal flow correction function. `cf[i]` is the one-qubit set correcting the measurement of qubit `i`.
    layers : list[set[int]]
        Partial order between corrected qubits in a layer form. The set `layers[i]` comprises the nodes in layer `i`. Nodes in layer `i` are "larger" in the partial order than nodes in layer `i+1`.


    Returns
    -------
    CausalFlow | None
        A causal flow object if the open graph has causal flow, `None` otherwise.
    """
    corrected_nodes_new: set[int] = set()
    corrector_nodes_new: set[int] = set()
    curr_layer: set[int] = set()

    non_corrected_nodes = og.graph.nodes - corrected_nodes

    if corrected_nodes == set(og.graph.nodes):
        return CausalFlow(og, cf, layers)

    for p in corrector_candidates:
        non_corrected_neighbors = og.neighbors({p}) & non_corrected_nodes
        if len(non_corrected_neighbors) == 1:
            (q,) = non_corrected_neighbors
            cf[q] = {p}
            curr_layer.add(q)
            corrected_nodes_new |= {q}
            corrector_nodes_new |= {p}

    layers.append(curr_layer)

    if len(corrected_nodes_new) == 0:
        return None

    corrected_nodes |= corrected_nodes_new
    corrector_candidates = (corrector_candidates - corrector_nodes_new) | (corrected_nodes_new & non_input_nodes)

    return _flow_aux(og, non_input_nodes, corrected_nodes, corrector_candidates, cf, layers)
