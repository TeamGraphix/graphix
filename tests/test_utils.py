from __future__ import annotations

from typing import Literal

import pytest

from graphix.utils import extract_qubits


def test_extract_qubits() -> None:
    assert extract_qubits((0, 1), (0, 0)) == (0, 1)
    with pytest.raises(ValueError, match="qubits expected"):
        extract_qubits(0, (0, 0))
    with pytest.raises(ValueError, match="qubits expected"):
        extract_qubits((0, 1, 2), (0, 0))
    with pytest.raises(ValueError, match="not enough values to unpack"):
        # The following line contains a type error:
        # Need more than 2 values to unpack (3 expected)  [misc]
        # If mypy does not catch it, a warning will be issued (Unused
        # "type: ignore" comment).
        _a, _b, _c = extract_qubits((0, 1), (0, 0))  # type: ignore[misc]
    # Check unsafe use
    s: tuple[Literal[0]] = (0,)
    _x: tuple[Literal[0]] = extract_qubits((1,), s)
