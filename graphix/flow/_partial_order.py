"""Tools for computing the partial orders."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet


def compute_topological_generations(
    dag: Mapping[int, AbstractSet[int]], indegree_map: Mapping[int, int], zero_indegree: AbstractSet[int]
) -> tuple[frozenset[int], ...] | None:
    """Stratify the directed acyclic graph (DAG) into generations.

    Parameters
    ----------
    dag : Mapping[int, AbstractSet[int]]
        Mapping encoding the directed acyclic graph.

    indegree_map : Mapping[int, int]
        Indegree of the input DAG. A pair (``key``, ``value``) represents a node in the DAG and the number of incoming edges incident on it. It is assumed that indegree values are larger than 0.

    zero_indegree : AbstractSet[int]
        Nodes in the DAG without any incoming edges.

    Returns
    -------
    tuple[frozenset[int], ...] | None
        Topological generations. `None` if the input DAG contains closed loops.

    Notes
    -----
    This function is adapted from `:func: networkx.algorithms.dag.topological_generations` so that it works directly on dictionaries instead of a `:class: nx.DiGraph` object.
    """
    generations: list[frozenset[int]] = []
    indegree_map = dict(indegree_map)

    while zero_indegree:
        this_generation = zero_indegree
        zero_indegree = set()
        for node in this_generation:
            for child in dag[node]:
                indegree_map[child] -= 1
                if indegree_map[child] == 0:
                    zero_indegree.add(child)
                    del indegree_map[child]
        generations.append(frozenset(this_generation))

    if indegree_map:
        return None

    return tuple(generations)
