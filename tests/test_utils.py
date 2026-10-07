import pytest

from graphix.utils import extract_qubits

def test_extract_qubits() -> None:
    assert extract_qubits((0, 1), (0, 0)) == (0, 1)
    with pytest.raises(ValueError, match="qubits expected"):
        extract_qubits(0, (0, 0))
    with pytest.raises(ValueError, match="qubits expected"):
        extract_qubits((0, 1, 2), (0, 0))
