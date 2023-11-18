"""Graph simulator

Graph state simulator, according to
M. Elliot, B. Eastin & C. Caves,
    JPhysA 43, 025301 (2010) and PRA 77, 042307 (2008)

"""
from __future__ import annotations

import warnings

RUSTWORKX_INSTALLED = False
try:
    import rustworkx as rx

    RUSTWORKX_INSTALLED = True
except ImportError:
    rx = None

from .nx_graphstate import NetworkxGraphState
from .rx_graphstate import RustworkxGraphState


class GraphState:
    """Factory class for graph state simulator."""

    def __new__(self, nodes=None, edges=None, vops=None, use_rustworkx: bool = True):
        if use_rustworkx:
            if RUSTWORKX_INSTALLED:
                return RustworkxGraphState(nodes=nodes, edges=edges, vops=vops)
            else:
                warnings.warn("rustworkx is not installed. Using networkx instead.")
        return NetworkxGraphState(nodes=nodes, edges=edges, vops=vops)
