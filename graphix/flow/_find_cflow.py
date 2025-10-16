from __future__ import annotations

from typing import TYPE_CHECKING

from graphix.flow.flow import CausalFlow
from graphix.fundamentals import Plane

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet

    from graphix.opengraph_ import OpenGraph

# TODO: Up doc strings


def find_cflow(og: OpenGraph[Plane]) -> CausalFlow | None:
    """Return the causal flow of the input open graph if it exists.

    Parameters
    ----------
    og : OpenGraph[Plane]
        Open graph whose causal flow is calculated.

    Returns
    -------
    cf : dict[int, set[int]]
        Causal flow correction function. `cf[i]` is the one-qubit set correcting the measurement of qubit `i`.
    layers : list[set[int]]
        Partial order between corrected qubits in a layer form. In particular, the set `layers[i]` comprises the nodes in layer `i`. Nodes in layer 0 are the "largest" nodes in the partial order.

    or `None`
        if the input open graph does not have a causal flow.

    Notes
    -----
    See Definition 2, Theorem 1 and Algorithm 1 in Mhalla and Perdrix, Finding Optimal Flows Efficiently, 2008 (arXiv:0709.2670).
    """
    if {Plane.XZ, Plane.YZ}.intersection(og.measurements.values()):
        return None

    corrected_nodes = set(og.output_nodes)
    corrector_candidates = corrected_nodes - set(og.input_nodes)

    cf: dict[int, set[int]] = {}
    layers: list[set[int]] = [corrected_nodes]

    non_input_nodes = og.graph.nodes - set(og.input_nodes)

    return _flow_aux(og, non_input_nodes, corrected_nodes, corrector_candidates, cf, layers)


def _flow_aux(
    og: OpenGraph[Plane],
    non_input_nodes: AbstractSet[int],
    corrected_nodes: AbstractSet[int],
    corrector_candidates: AbstractSet[int],
    cf: dict[int, set[int]],
    layers: list[set[int]],
) -> CausalFlow | None:
    """Find one layer of the causal flow.

    Parameters
    ----------
    og : OpenGraph[Plane]
        Open graph whose causal flow is calculated.
    non_input_nodes : AbstractSet[int]
        Non-input nodes of the input open graph. This parameter remains constant throughout the execution of the algorithm and can be derived from `og` at any time. It is passed as an argument to avoid unnecessary recalulations.
    corrected_nodes : AbstractSet[int]
        Nodes which have already been corrected.
    corrector_candidates : AbstractSet[int]
        Nodes which could correct a node at the time of calling the function. This set can never contain input nodes, uncorrected nodes or nodes which already correct another node.
    cf : dict[int, set[int]]
        Causal flow correction function. `cf[i]` is the one-qubit set correcting the measurement of qubit `i`.
    layers : list[set[int]]
        Partial order between corrected qubits in a layer form. In particular, the set `layers[i]` comprises the nodes in layer `i`. Nodes in layer 0 are the "largest" nodes in the partial order.


    Returns
    -------
    cf : dict[int, set[int]]
        Causal flow correction function. `cf[i]` is the one-qubit set correcting the measurement of qubit `i`.
    layers : list[set[int]]
        Partial order between corrected qubits in a layer form. In particular, the set `layers[i]` comprises the nodes in layer `i`. Nodes in layer 0 are the "largest" nodes in the partial order.

    or `None`
        if the input open graph does not have a causal flow.
    """
    corrected_nodes_new: set[int] = set()
    corrector_nodes_new: set[int] = set()
    curr_layer: set[int] = set()

    non_corrected_nodes = og.graph.nodes - corrected_nodes

    for p in corrector_candidates:
        non_corrected_neighbors = og.neighbors({p}) & non_corrected_nodes
        if len(non_corrected_neighbors) == 1:
            (q,) = non_corrected_neighbors
            cf[q] = {p}
            curr_layer.add(p)
            corrected_nodes_new |= {q}
            corrector_nodes_new |= {p}

    layers.append(curr_layer)

    if len(corrected_nodes_new) == 0:
        # TODO: This is the structure in the original graphix code. I think that we could check if non_corrected_nodes == empty before the loop and here just return None.
        if corrected_nodes == og.graph.nodes:
            return CausalFlow(og, cf, layers)
        return None

    corrected_nodes |= corrected_nodes_new
    corrector_candidates = (corrector_candidates - corrector_nodes_new) | (corrected_nodes_new & non_input_nodes)

    return _flow_aux(og, non_input_nodes, corrected_nodes, corrector_candidates, cf, layers)
