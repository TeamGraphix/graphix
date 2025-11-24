"""Data structure for single-qubit measurements in MBQC."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import (
    Literal,
    NamedTuple,
    SupportsInt,
    TypeAlias,
)

from graphix import utils
from graphix.fundamentals import AbstractPlanarMeasurement, Axis, Plane, Sign

# Ruff suggests to move this import to a type-checking block, but dataclass requires it here
from graphix.parameter import ExpressionOrFloat  # noqa: TC001

Outcome: TypeAlias = Literal[0, 1]


def outcome(b: bool) -> Outcome:
    """Return 1 if True, 0 if False."""
    return 1 if b else 0


def toggle_outcome(outcome: Outcome) -> Outcome:
    """Toggle outcome."""
    return 1 if outcome == 0 else 0


@dataclass
class Domains:
    """Represent `X^sZ^t` where s and t are XOR of results from given sets of indices."""

    s_domain: set[int]
    t_domain: set[int]


@dataclass
class Measurement(AbstractPlanarMeasurement):
    r"""An MBQC measurement.

    Attributes
    ----------
    angle : Expressionor Float
        The angle of the measurement in units of :math:`\pi`. Should be between [0, 2).
    plane : graphix.fundamentals.Plane
        The measurement plane.
    """

    angle: ExpressionOrFloat
    plane: Plane

    def isclose(self, other: Measurement, rel_tol: float = 1e-09, abs_tol: float = 0.0) -> bool:
        """Compare if two measurements have the same plane and their angles are close.

        Example
        -------
        >>> from graphix.measurements import Measurement
        >>> from graphix.fundamentals import Plane
        >>> Measurement(0.0, Plane.XY).isclose(Measurement(0.0, Plane.XY))
        True
        >>> Measurement(0.0, Plane.XY).isclose(Measurement(0.0, Plane.YZ))
        False
        >>> Measurement(0.1, Plane.XY).isclose(Measurement(0.0, Plane.XY))
        False
        """
        return (
            math.isclose(self.angle, other.angle, rel_tol=rel_tol, abs_tol=abs_tol)
            if isinstance(self.angle, float) and isinstance(other.angle, float)
            else self.angle == other.angle
        ) and self.plane == other.plane

    def to_plane_or_axis(self) -> Plane | Axis:
        """Return the measurements's plane or axis.

        Returns
        -------
        Plane | Axis

        Notes
        -----
        Measurements with Pauli angles (i.e., ``self.angle == n/2`` with ``n`` an integer) are interpreted as `Axis` instances.
        """
        if pm := PauliMeasurement.try_from(self.plane, self.angle):
            return pm.axis
        return self.plane

    def to_plane(self) -> Plane:
        """Return the measurement's plane.

        Returns
        -------
        Plane
        """
        return self.plane


class PauliMeasurement(NamedTuple):
    """Pauli measurement."""

    axis: Axis
    sign: Sign

    @staticmethod
    def try_from(plane: Plane, angle: ExpressionOrFloat) -> PauliMeasurement | None:
        """Return the Pauli measurement description if a given measure is Pauli."""
        angle_double = 2 * angle
        if not isinstance(angle_double, SupportsInt) or not utils.is_integer(angle_double):
            return None
        angle_double_mod_4 = int(angle_double) % 4
        axis = plane.cos if angle_double_mod_4 % 2 == 0 else plane.sin
        sign = Sign.minus_if(angle_double_mod_4 >= 2)
        return PauliMeasurement(axis, sign)
