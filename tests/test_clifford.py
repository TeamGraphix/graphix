from __future__ import annotations

from graphix.clifford import Clifford


class TestClifford:
    def test_named(self) -> None:
        assert hasattr(Clifford, "I")
        assert hasattr(Clifford, "X")
        assert hasattr(Clifford, "Y")
        assert hasattr(Clifford, "Z")
        assert hasattr(Clifford, "S")
        assert hasattr(Clifford, "H")

    def test_iteration(self) -> None:
        """Test that Clifford iteration does not take (I, X, Y, Z, S, H) into account."""
        assert len(Clifford) == 24
        assert len(frozenset(Clifford)) == 24

    def test_index_type(self) -> None:
        for c in Clifford:
            assert isinstance(c.index, int)
