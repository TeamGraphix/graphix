from __future__ import annotations

from copy import deepcopy
from enum import Enum

import numpy as np

from graphix.graphsim.basegraphstate import BaseGraphState
from graphix.graphsim.graphstate import GraphState
from graphix.graphsim.rxgraphstate import RXGraphState
from graphix.graphsim.utils import is_graphs_equal


class ResourceType(Enum):
    GHZ = "GHZ"
    LINEAR = "LINEAR"
    NONE = None

    def __str__(self):
        return self.name


class ResourceGraph:
    """resource graph state object.

    Parameters
    ----------
    type : :class:`ResourceType` object
        Type of the cluster.
    graph : :class:`~graphix.graphsim.GraphState` object
        Graph state of the cluster.
    """

    def __init__(self, type: ResourceType, graph: GraphState = None):
        self.graph = graph
        self.type = type

    def __str__(self) -> str:
        return str(self.type) + str(self.graph.nodes)

    def __repr__(self) -> str:
        return str(self.type) + str(self.graph.nodes)

    def __eq__(self, other) -> bool:
        if not isinstance(other, ResourceGraph):
            raise TypeError("cannot compare ResourceGraph with other object")

        return is_graphs_equal(self.graph, other.graph) and self.type == other.type


def get_fusion_network_from_graph(
    graph: BaseGraphState,
    max_ghz: int | float = np.inf,
    max_lin: int | float = np.inf,
) -> list[ResourceGraph]:
    """Extract GHZ and linear cluster graph state decomposition of desired resource state :class:`~graphix.graphsim.GraphState`.
    Extraction algorithm is based on [1].

    [1] Zilk et al., A compiler for universal photonic quantum computers, 2022 `arXiv:2210.09251 <https://arxiv.org/abs/2210.09251>`_

    Parameters
    ----------
    graph : :class:`~graphix.graphsim.GraphState` object
        Graph state.
    phasedict : dict
        Dictionary of phases for each node.
    max_ghz:
        Maximum size of ghz clusters
    max_lin:
        Maximum size of linear clusters

    Returns
    -------
    list
        List of :class:`ResourceGraph` objects.
    """
    use_rustworkx = isinstance(graph, RXGraphState)

    adjdict = deepcopy({n: adj for n, adj in graph.adjacency()})

    number_of_edges = graph.number_of_edges()
    resource_list = []
    neighbors_list = []

    # Prepare a list sorted by number of neighbors to get the largest GHZ clusters first.
    for v in adjdict.keys():
        if len(adjdict[v]) > 2:
            neighbors_list.append((v, len(adjdict[v])))
        # If there is an isolated node, add it to the list.
        if len(adjdict[v]) == 0:
            resource_list.append(create_resource_graph([v], root=v, use_rustworkx=use_rustworkx))

    # Find GHZ graphs in the graph and remove their edges from the graph.
    # All nodes that have more than 2 edges become the roots of the GHZ clusters.
    for v, _ in sorted(neighbors_list, key=lambda tup: tup[1], reverse=True):
        if len(adjdict[v]) > 2:
            nodes = [v]
            while len(adjdict[v]) > 0 and len(nodes) < max_ghz:
                n, _ = adjdict[v].popitem()
                nodes.append(n)
                del adjdict[n][v]
                number_of_edges -= 1
            resource_list.append(create_resource_graph(nodes, root=v, use_rustworkx=use_rustworkx))

    # Find Linear clusters in the remaining graph and remove their edges from the graph.
    while number_of_edges != 0:
        for v in adjdict.keys():
            if len(adjdict[v]) == 1:
                n = v
                nodes = [n]
                while len(adjdict[n]) > 0 and len(nodes) < max_lin:
                    n2, _ = adjdict[n].popitem()
                    nodes.append(n2)
                    del adjdict[n2][n]
                    number_of_edges -= 1
                    n = n2

                # We define any cluster whose size is smaller than 4, a GHZ cluster
                if len(nodes) == 3:
                    resource_list.append(
                        create_resource_graph(
                            [nodes[1], nodes[0], nodes[2]], root=nodes[1], use_rustworkx=use_rustworkx
                        )
                    )
                elif len(nodes) == 2:
                    resource_list.append(create_resource_graph(nodes, root=nodes[0], use_rustworkx=use_rustworkx))
                else:
                    resource_list.append(create_resource_graph(nodes, use_rustworkx=use_rustworkx))

        # If a cycle exists in the graph, extract one 3-qubit ghz cluster from the cycle.
        for v in adjdict.keys():
            if len(adjdict[v]) == 2:
                neighbors = list(adjdict[v].keys())
                nodes = [v] + neighbors
                del adjdict[neighbors[0]][v]
                del adjdict[neighbors[1]][v]
                del adjdict[v][neighbors[0]]
                del adjdict[v][neighbors[1]]
                number_of_edges -= 2

                resource_list.append(create_resource_graph(nodes, root=v, use_rustworkx=use_rustworkx))
                break
    return resource_list


def create_resource_graph(node_ids: list[int], root: int | None = None, use_rustworkx=False) -> ResourceGraph:
    """Create a resource graph state (GHZ or linear) from node ids.

    Parameters
    ----------
    node_ids : list
        List of node ids.
    root : int
        Root of the ghz cluster. If None, it's a linear cluster.

    Returns
    -------
    :class:`Cluster` object
        Cluster object.
    """
    cluster_type = None
    edges = []
    if root is not None:
        edges = [(root, i) for i in node_ids if i != root]
        cluster_type = ResourceType.GHZ
    else:
        edges = [(node_ids[i], node_ids[i + 1]) for i in range(len(node_ids)) if i + 1 < len(node_ids)]
        cluster_type = ResourceType.LINEAR
    tmp_graph = GraphState(use_rustworkx=use_rustworkx)
    tmp_graph.add_nodes_from(node_ids)
    tmp_graph.add_edges_from(edges)
    return ResourceGraph(type=cluster_type, graph=tmp_graph)


def get_fusion_nodes(c1: ResourceGraph, c2: ResourceGraph) -> list[int]:
    """Get the nodes that are fused between two resource states. Currently, we consider only type-I fusion.
    See [2] for the definition of fusion operation.

    [2] Daniel E. Browne and Terry Rudolph. Resource-efficient linear optical quantum computation. Physical Review Letters, 95(1):010501, 2005.

    Parameters
    ----------
    c1 : :class:`ResourceGraph` object
        First resource state to be fused.
    c2 : :class:`ResourceGraph` object
        Second resource state to be fused.

    Returns
    -------
    list
        List of nodes that are fused between the two clusters.
    """
    if not isinstance(c1, ResourceGraph) or not isinstance(c2, ResourceGraph):
        raise TypeError("c1 and c2 must be Cluster objects")

    if c1 == c2:
        return []
    return [n for n in c1.graph.nodes if n in c2.graph.nodes]
