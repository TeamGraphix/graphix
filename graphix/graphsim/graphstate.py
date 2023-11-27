"""Graph simulator

Graph state simulator, according to
M. Elliot, B. Eastin & C. Caves,
    JPhysA 43, 025301 (2010) and PRA 77, 042307 (2008)

"""
from __future__ import annotations

import warnings

from .basegraphstate import RUSTWORKX_INSTALLED, BaseGraphState
from .nxgraphstate import NXGraphState
from .rxgraphstate import RXGraphState


class GraphState:
    """Factory class for graph state simulator."""

    def __new__(self, nodes=None, edges=None, vops=None, use_rustworkx: bool = False) -> BaseGraphState:
        if use_rustworkx:
            if RUSTWORKX_INSTALLED:
                return RXGraphState(nodes=nodes, edges=edges, vops=vops)
            else:
                warnings.warn("rustworkx is not installed. Using networkx instead.")
        return NXGraphState(nodes=nodes, edges=edges, vops=vops)
