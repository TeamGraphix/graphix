from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from graphix import Clifford
from graphix.command import BaseM, BaseN, Command

if TYPE_CHECKING:
    from graphix.command import BaseCommand


@pytest.mark.parametrize(
    ("cmd1", "cmd2"),
    [
        (BaseN(0), BaseN(1)),
        (BaseN(2), BaseN(2)),
        (Command.N(0), Command.N(1)),
        (Command.N(2), Command.N(2)),
        (BaseM(0), BaseM(1)),
        (BaseM(2), BaseM(2)),
        (Command.M(0), Command.M(1)),
        (Command.M(2), Command.M(2)),
        (Command.E((0, 2)), Command.E((1, 2))),
        (Command.E((2, 0)), Command.E((2, 1))),
        (Command.C(0, Clifford.H), Command.C(1, Clifford.H)),
        (Command.C(2, Clifford.S), Command.C(2, Clifford.S)),
        (Command.X(0, {2}), Command.X(1, {2})),
        (Command.X(2, {0}), Command.X(2, {1})),
        (Command.Z(0, {2}), Command.Z(1, {2})),
        (Command.Z(2, {0}), Command.Z(2, {1})),
        (Command.S(0, {2}), Command.S(1, {2})),
        (Command.S(2, {0}), Command.S(2, {1})),
        (Command.T(), Command.T()),
    ],
)
def test_reindex(cmd1: BaseCommand, cmd2: BaseCommand) -> None:
    def reindex(node: int) -> int:
        return 1 if node == 0 else node

    assert cmd1.reindex(reindex) == cmd2
