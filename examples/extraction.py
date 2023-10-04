"""
Extract clusters from :class:`~graphix.GraphState`.
=======================================================

In this example, we extract GHZ clusters and linear clusters from :class:`~graphix.GraphState`.
The extraction algorithm is based on [1].

[1] Zilk et al., A compiler for universal photonic quantum computers, 2022 `arXiv:2210.09251 <https://arxiv.org/abs/2210.09251>`_

"""

import itertools
import graphix
from graphix.extraction import extract_clusters_from_graph

# %%
# Here we create a graph state with 9 nodes and 12 edges.
gs = graphix.GraphState()
nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (0, 5), (4, 5), (5, 6), (6, 7), (7, 0), (7, 8), (8, 1)]
gs.add_nodes_from(nodes)
gs.add_edges_from(edges)
gs.draw()

# %%
# Extracted GHZ clusters and linear clusters from the graph state are shown below.
extract_clusters_from_graph(gs)

# %%
# If you want to know what nodes are fused in each clusters, you can use :func:`~graphix.extraction.get_fusion_nodes` function.
# Currently, we consider only type-I fusion. See [2] for the definition of fusion.
#
# [2] Daniel E. Browne and Terry Rudolph. Resource-efficient linear optical quantum computation. Physical Review Letters, 95(1):010501, 2005.
clusters = extract_clusters_from_graph(gs)
for idx1, idx2 in itertools.combinations(range(len(clusters)), 2):
    print(
        f"fusion nodes between cluster {idx1} and cluster {idx2}: {graphix.extraction.get_fusion_nodes(clusters[idx1], clusters[idx2])}"
    )

# %%
# You can also specify the maximum size of GHZ clusters and linear clusters.
extract_clusters_from_graph(gs, max_ghz=4, max_lin=4)
