from __future__ import annotations

import pytest

from graphix.clifford import Clifford
from graphix.fundamentals import Plane
from graphix.measurements import Measurement, PauliMeasurement


class TestMeasurement:
    def test_isclose(self) -> None:
        m1 = Measurement.XY(0.1)
        m2 = Measurement.XY(0.15)

        assert not m1.isclose(m2)
        assert not m1.isclose(Plane.XY)
        assert m1.isclose(m2, abs_tol=0.1)


@pytest.mark.parametrize("pauli", PauliMeasurement)
def test_pauli_to_bloch(pauli: PauliMeasurement) -> None:
    bloch = pauli.to_bloch()
    pauli_back = bloch.try_to_pauli()
    assert pauli == pauli_back


@pytest.mark.parametrize("pauli", PauliMeasurement)
@pytest.mark.parametrize("clifford", Clifford)
def test_clifford_pauli(pauli: PauliMeasurement, clifford: Clifford) -> None:
    pauli_clifford = pauli.clifford(clifford)
    bloch = pauli.to_bloch()
    bloch_clifford = bloch.clifford(clifford)
    assert pauli_clifford == bloch_clifford.try_to_pauli()


def test_clifford() -> None:
    assert Measurement.XY(0.25).clifford(Clifford.H @ Clifford.S @ Clifford.Z) == Measurement.XZ(0.25)
