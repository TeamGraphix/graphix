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

# override introduced in Python 3.12
from typing_extensions import override

from graphix import utils
from graphix.fundamentals import AbstractMeasurement, AbstractPlanarMeasurement, Axis, Plane, Sign

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
    angle : ExpressionOrFloat
        The angle of the measurement in units of :math:`\pi`. Should be between [0, 2).
    plane : graphix.fundamentals.Plane
        The measurement plane.
    """

    angle: ExpressionOrFloat
    plane: Plane

    @override
    def isclose(self, other: AbstractMeasurement, rel_tol: float = 1e-09, abs_tol: float = 0.0) -> bool:
        """Determine whether two measurements are close in angle and share the same plane.

        This method compares the angle of the current measurement with that of
        another measurement, using :func:`math.isclose` when both angles are floats.
        The planes must match exactly for the measurements to be considered close.

        Parameters
        ----------
        other : AbstractMeasurement
            The measurement to compare against.
        rel_tol : float, optional
            Relative tolerance for comparing angles, passed to :func:`math.isclose`. Default is ``1e-9``.
        abs_tol : float, optional
            Absolute tolerance for comparing angles, passed to :func:`math.isclose`. Default is ``0.0``.

        Returns
        -------
        bool
        ``True`` if both measurements lie in the same plane and their angles
        are equal or close within the given tolerances; ``False`` otherwise.

        Examples
        --------
        >>> from graphix.measurements import Measurement
        >>> from graphix.fundamentals import Plane
        >>> Measurement(0.0, Plane.XY).isclose(Measurement(0.0, Plane.XY))
        True
        >>> Measurement(0.0, Plane.XY).isclose(Measurement(0.0, Plane.YZ))
        False
        >>> Measurement(0.1, Plane.XY).isclose(Measurement(0.0, Plane.XY))
        False
        """
        if isinstance(other, Measurement):
            return (
                math.isclose(self.angle, other.angle, rel_tol=rel_tol, abs_tol=abs_tol)
                if isinstance(self.angle, float) and isinstance(other.angle, float)
                else self.angle == other.angle
            ) and self.plane == other.plane
        return False

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
