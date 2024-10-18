"""Graph simulator.

Graph state simulator, according to
M. Elliot, B. Eastin & C. Caves,
    JPhysA 43, 025301 (2010) and PRA 77, 042307 (2008)

"""

from __future__ import annotations

import warnings

from graphix.graphsim.basegraphstate import RUSTWORKX_INSTALLED, BaseGraphState
from graphix.graphsim.nxgraphstate import NXGraphState
from graphix.graphsim.rxgraphstate import RXGraphState


class GraphState:
    """Factory class for graph state simulator."""

    def __new__(cls, nodes=None, edges=None, vops=None, use_rustworkx: bool = False) -> BaseGraphState:
        """Build a graph state simulator."""
        if use_rustworkx:
            if RUSTWORKX_INSTALLED:
                return RXGraphState(nodes=nodes, edges=edges, vops=vops)
            else:
                warnings.warn("rustworkx is not installed. Using networkx instead.", stacklevel=1)
        return NXGraphState(nodes=nodes, edges=edges, vops=vops)
