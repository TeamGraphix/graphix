from __future__ import annotations
from enum import Enum

import graphix
import numpy as np
import networkx as nx


class ClusterType(Enum):
    GHZ = "GHZ"
    LINEAR = "LINEAR"
    NONE = None

    def __str__(self):
        return self.name


class Cluster:
    """Cluster object.

    Parameters
    ----------
    type : :class:`ClusterType` object
        Type of the cluster.
    graph : :class:`graphix.GraphState` object
        Graph state of the cluster.
    """

    def __init__(self, type: ClusterType, graph: graphix.GraphState = None):
        self.graph = graph
        self.type = type

    def __str__(self) -> str:
        return str(self.type) + str(self.graph.nodes)

    def __repr__(self) -> str:
        return str(self.type) + str(self.graph.nodes)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Cluster):
            raise TypeError("cannot compare Cluster with non-Cluster object")

        return nx.utils.graphs_equal(self.graph, other.graph) and self.type == other.type


def extract_clusters_from_graph(
    graph: graphix.GraphState,
    max_ghz: int | float = np.inf,
    max_lin: int | float = np.inf,
) -> list[Cluster]:
    """Extract GHZ clusters and Linear clusters circuit from :class:`graphix.GraphState`.
    Extraction algorithm is based on [1].

    [1] Zilk et al., A compiler for universal photonic quantum computers, 2022 `arXiv:2210.09251 <https://arxiv.org/abs/2210.09251>`_

    Parameters
    ----------
    graph : :class:`graphix.GraphState` object
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
        List of :class:`Cluster` objects.
    """
    adjdict = nx.to_dict_of_dicts(graph)
    number_of_edges = graph.number_of_edges()
    cluster_list = []
    neighbors_list = []

    # Prepare a list sorted by number of neighbors to get the largest GHZ clusters first.
    for v in adjdict.keys():
        if len(adjdict[v]) > 2:
            neighbors_list.append((v, len(adjdict[v])))
        # If there is an isolated node, add it to the list.
        if len(adjdict[v]) == 0:
            cluster_list.append(create_cluster([v], root=v))

    # Find GHZ clusters in the graph and remove their edges from the graph.
    # All nodes that have more than 2 edges become the roots of the GHZ clusters.
    for v, _ in sorted(neighbors_list, key=lambda tup: tup[1], reverse=True):
        if len(adjdict[v]) > 2:
            nodes = [v]
            while len(adjdict[v]) > 0 and len(nodes) < max_ghz:
                n, _ = adjdict[v].popitem()
                nodes.append(n)
                del adjdict[n][v]
                number_of_edges -= 1
            cluster_list.append(create_cluster(nodes, root=v))

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
                    cluster_list.append(create_cluster([nodes[1], nodes[0], nodes[2]], root=nodes[1]))
                elif len(nodes) == 2:
                    cluster_list.append(create_cluster(nodes, root=nodes[0]))
                else:
                    cluster_list.append(create_cluster(nodes))

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

                cluster_list.append(create_cluster(nodes, root=v))
                break
    return cluster_list


def create_cluster(node_ids: list[int], root: int | None = None) -> Cluster:
    """Create a cluster from node ids.

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
        cluster_type = ClusterType.GHZ
    else:
        edges = [(node_ids[i], node_ids[i + 1]) for i in range(len(node_ids)) if i + 1 < len(node_ids)]
        cluster_type = ClusterType.LINEAR
    tmp_graph = graphix.GraphState()
    tmp_graph.add_nodes_from(node_ids)
    tmp_graph.add_edges_from(edges)
    return Cluster(type=cluster_type, graph=tmp_graph)


def get_fusion_nodes(c1: Cluster, c2: Cluster) -> list[int]:
    """Get the nodes that are fused between two clusters.

    Parameters
    ----------
    c1 : :class:`Cluster` object
        First cluster.
    c2 : :class:`Cluster` object
        Second cluster.

    Returns
    -------
    list
        List of nodes that are fused between the two clusters.
    """
    if not isinstance(c1, Cluster) or not isinstance(c2, Cluster):
        raise TypeError("c1 and c2 must be Cluster objects")

    if c1 == c2:
        return []
    return [n for n in c1.graph.nodes if n in c2.graph.nodes]
