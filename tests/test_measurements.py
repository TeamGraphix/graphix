from __future__ import annotations

from graphix.fundamentals import Plane
from graphix.measurements import Measurement


class TestMeasurement:
    def test_isclose(self) -> None:
        m1 = Measurement(0.1, Plane.XY)
        m2 = Measurement(0.15, Plane.XY)

        assert not m1.isclose(m2)
        assert not m1.isclose(Plane.XY)
        assert m1.isclose(m2, abs_tol=0.1)
