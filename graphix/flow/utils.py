"""Module for flow utils."""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet

    import networkx as nx


def find_odd_neighbor(graph: nx.Graph[int], vertices: AbstractSet[int]) -> set[int]:
    """Return the odd neighborhood of a set of nodes.

    Parameters
    ----------
    graph : networkx.Graph
        Underlying graph.
    vertices : set
        Set of nodes of which to find the odd neighborhood.

    Returns
    -------
    odd_neighbors : set
        Set of indices for odd neighbor of set `vertices`.
    """
    odd_neighbors: set[int] = set()
    for vertex in vertices:
        neighbors = set(graph.neighbors(vertex))
        odd_neighbors ^= neighbors
    return odd_neighbors
