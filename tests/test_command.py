from __future__ import annotations

from graphix import Clifford
from graphix.command import BaseM, BaseN, Command


def test_reindex() -> None:
    def reindex(node: int) -> int:
        return 1 if node == 0 else node

    assert BaseN(0).reindex(reindex) == BaseN(1)
    assert BaseN(2).reindex(reindex) == BaseN(2)
    assert Command.N(0).reindex(reindex) == Command.N(1)
    assert Command.N(2).reindex(reindex) == Command.N(2)
    assert BaseM(0).reindex(reindex) == BaseM(1)
    assert BaseM(2).reindex(reindex) == BaseM(2)
    assert Command.M(0).reindex(reindex) == Command.M(1)
    assert Command.M(2).reindex(reindex) == Command.M(2)
    assert Command.E((0, 2)).reindex(reindex) == Command.E((1, 2))
    assert Command.E((2, 0)).reindex(reindex) == Command.E((2, 1))
    assert Command.C(0, Clifford.H).reindex(reindex) == Command.C(1, Clifford.H)
    assert Command.C(2, Clifford.S).reindex(reindex) == Command.C(2, Clifford.S)
    assert Command.X(0, {2}).reindex(reindex) == Command.X(1, {2})
    assert Command.X(2, {0}).reindex(reindex) == Command.X(2, {1})
    assert Command.Z(0, {2}).reindex(reindex) == Command.Z(1, {2})
    assert Command.Z(2, {0}).reindex(reindex) == Command.Z(2, {1})
    assert Command.S(0, {2}).reindex(reindex) == Command.S(1, {2})
    assert Command.S(2, {0}).reindex(reindex) == Command.S(2, {1})
    assert Command.T().reindex(reindex) == Command.T()
