"""MBQC measurements."""

from __future__ import annotations

import dataclasses
import math

from graphix import type_utils
from graphix.pauli import Axis, Plane, Sign


@dataclasses.dataclass
class Domains:
    """Represent `X^sZ^t` where s and t are XOR of results from given sets of indices."""

    s_domain: set[int]
    t_domain: set[int]


@dataclasses.dataclass(frozen=True)
class Measurement:
    """An MBQC measurement.

    :param angle: the angle of the measurement. Should be between [0, 2)
    :param plane: the measurement plane
    """

    angle: float
    plane: Plane

    def isclose(self, other: Measurement, rel_tol: float = 1e-09, abs_tol: float = 0.0) -> bool:
        """Compare if two measurements have the same plane and their angles are close.

        Example
        -------
        >>> from graphix.opengraph import Measurement
        >>> from graphix.pauli import Plane
        >>> Measurement(0.0, Plane.XY).isclose(Measurement(0.0, Plane.XY))
        True
        >>> Measurement(0.0, Plane.XY).isclose(Measurement(0.0, Plane.YZ))
        False
        >>> Measurement(0.1, Plane.XY).isclose(Measurement(0.0, Plane.XY))
        False
        """
        return math.isclose(self.angle, other.angle, rel_tol=rel_tol, abs_tol=abs_tol) and self.plane == other.plane


@dataclasses.dataclass(frozen=True)
class PauliMeasurement:
    """Pauli measurement."""

    axis: Axis
    sign: Sign

    @staticmethod
    def try_from(plane: Plane, angle: float) -> PauliMeasurement | None:
        """Return the Pauli measurement description if a given measure is Pauli."""
        angle_double = 2 * angle
        if not type_utils.is_integer(angle_double):
            return None
        angle_double_mod_4 = int(angle_double) % 4
        axis = plane.cos if angle_double_mod_4 % 2 == 0 else plane.sin
        sign = Sign.minus_if(angle_double_mod_4 >= 2)
        return PauliMeasurement(axis, sign)
