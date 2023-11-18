"""Graph simulator

Graph state simulator, according to
M. Elliot, B. Eastin & C. Caves,
    JPhysA 43, 025301 (2010) and PRA 77, 042307 (2008)

"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Union

import networkx as nx

from graphix.clifford import CLIFFORD_HSZ_DECOMPOSITION, CLIFFORD_MUL
from graphix.ops import Ops
from graphix.sim.statevec import Statevec

from .nx_graphstate import NetworkxGraphState
from .rx_graphstate import RustworkxGraphState

RUSTWORKX_INSTALLED = False
try:
    import rustworkx as rx

    RUSTWORKX_INSTALLED = True
except ImportError:
    rx = None


GraphObject: Union[nx.Graph, rx.PyGraph]


class GraphState:
    """Factory class for graph state simulator."""

    def __new__(self, nodes=None, edges=None, vops=None, use_rustworkx: bool = True):
        if use_rustworkx:
            if RUSTWORKX_INSTALLED:
                return RustworkxGraphState(nodes=nodes, edges=edges, vops=vops)
            else:
                warnings.warn("rustworkx is not installed. Using networkx instead.")
        return NetworkxGraphState(nodes=nodes, edges=edges, vops=vops)
