"""Graph simulator

Graph state simulator, according to
M. Elliot, B. Eastin & C. Caves,
    JPhysA 43, 025301 (2010) and PRA 77, 042307 (2008)

"""

from __future__ import annotations

from .basegraphstate import BaseGraphState
from .nxgraphstate import NXGraphState


class GraphState:
    """Factory class for graph state simulator."""

    def __new__(
        cls,
        nodes: list[int] | None = None,
        edges: list[tuple[int, int]] | None = None,
        vops: dict[int, int] | None = None,
        use_rustworkx: bool = False,
    ) -> BaseGraphState:
        if use_rustworkx:
            from graphix.graphsim.rxgraphstate import RXGraphState

            return RXGraphState(nodes=nodes, edges=edges, vops=vops)
        return NXGraphState(nodes=nodes, edges=edges, vops=vops)
