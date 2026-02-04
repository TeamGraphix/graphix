"""Data structure for single-qubit measurements in MBQC."""

from __future__ import annotations

import math
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Literal,
)

# override introduced in Python 3.12
from typing_extensions import override

from graphix import parameter
from graphix.fundamentals import (
    ANGLE_PI,
    AbstractMeasurement,
    AbstractPlanarMeasurement,
    Angle,
    Axis,
    ParameterizedAngle,
    Plane,
    Sign,
)
from graphix.pauli import Pauli

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from typing import Self, TypeAlias

    from graphix.clifford import Clifford
    from graphix.parameter import ExpressionOrSupportsFloat, Parameter

Outcome: TypeAlias = Literal[0, 1]


def outcome(b: bool) -> Outcome:
    """Return 1 if True, 0 if False."""
    return 1 if b else 0


def toggle_outcome(outcome: Outcome) -> Outcome:
    """Toggle outcome."""
    return 1 if outcome == 0 else 0


@dataclass(frozen=True)
class Measurement(AbstractMeasurement):
    r"""An MBQC measurement.

    Base class for :class:`BlochMeasurement` and :class:`PauliMeasurement`.

    This class contains three class variables ``X``, ``Y``, and ``Z``
    for the positive Pauli measurements on the three axes, and three
    static methods ``XY``, ``YZ``, and ``XZ``, each parameterized by
    an angle and returning a Bloch measurement on each of the three
    planes.

    The three static methods ``XY``, ``YZ``, and ``XZ`` are
    capitalized, contrary to what PEP 8 prescribes, to remain
    consistent with the names of the Pauli measurements and to match
    the names of the planes in the :class:`Plane` enum.

    """

    # The actual values for the class variables ``X``, ``Y``, and
    # ``Z`` are assigned latter in this file, once
    # ``PauliMeasurement`` is defined.
    X: ClassVar[PauliMeasurement]
    Y: ClassVar[PauliMeasurement]
    Z: ClassVar[PauliMeasurement]

    @staticmethod
    def XY(angle: ParameterizedAngle) -> BlochMeasurement:  # noqa: N802
        """Return a Bloch measurement on the XY plane."""
        return BlochMeasurement(angle, Plane.XY)

    @staticmethod
    def YZ(angle: ParameterizedAngle) -> BlochMeasurement:  # noqa: N802
        """Return a Bloch measurement on the YZ plane."""
        return BlochMeasurement(angle, Plane.YZ)

    @staticmethod
    def XZ(angle: ParameterizedAngle) -> BlochMeasurement:  # noqa: N802
        """Return a Bloch measurement on the XZ plane."""
        return BlochMeasurement(angle, Plane.XZ)

    @abstractmethod
    def clifford(self, clifford_gate: Clifford) -> Self:
        r"""Return a new measurement command with a :class:`Clifford` applied.

        Parameters
        ----------
        clifford_gate : Clifford
            Clifford gate to apply before the measurement.

        Returns
        -------
        Self
            Equivalent measurement representing the pattern ``MC``.
        """

    @abstractmethod
    def to_bloch(self) -> BlochMeasurement:
        """Return the measurement description as an angle and a plane on the Bloch sphere.

        There is no unique Bloch representation for each Pauli measurement.
        For instance,

            >>> from graphix.measurements import Measurement
            >>> Measurement.XY(0.5).try_to_pauli() == Measurement.YZ(0.5).try_to_pauli() == Measurement.Y
            True


        This method follows the convention illustrated below:

            >>> from graphix.measurements import PauliMeasurement
            >>> for pm in list(PauliMeasurement):
            ...     print(f"{pm}.to_bloch() == {pm.to_bloch()}")
            +X.to_bloch() == Measurement.XY(0)
            -X.to_bloch() == Measurement.XY(1)
            +Y.to_bloch() == Measurement.XY(0.5)
            -Y.to_bloch() == Measurement.XY(1.5)
            +Z.to_bloch() == Measurement.YZ(0)
            -Z.to_bloch() == Measurement.YZ(1)
        """

    @abstractmethod
    def downcast_bloch(self) -> BlochMeasurement:
        """Return the measurement description if it is already given as an angle and a plane on the Bloch sphere; raise :class:`TypeError` otherwise."""

    @abstractmethod
    def try_to_pauli(self, rel_tol: float = 1e-09, abs_tol: float = 0.0) -> PauliMeasurement | None:
        """Return the measurement description as a Pauli measurement if possible, or ``None`` otherwise.

        Parameters
        ----------
        rel_tol : float, optional
            Relative tolerance for comparing angles, passed to :func:`math.isclose`.
            Default is ``1e-9``.
        abs_tol : float, optional
            Absolute tolerance for comparing angles, passed to :func:`math.isclose`.
            Default is ``0.0``.

        Returns
        -------
        PauliMeasurement | None
            If ``self`` is already an instance of :class:`PauliMeasurement`, the function
            returns ``self``. If ``self`` is an instance of :class:`BlochMeasurement`, then
            either the measurement is close to a Pauli measurement (i.e., the angle is close to an
            integer multiple of π/2) and the corresponding Pauli measurement is returned,
            or it is not and ``None`` is returned.
        """

    @abstractmethod
    def to_pauli_or_bloch(self, rel_tol: float = 1e-09, abs_tol: float = 0.0) -> PauliMeasurement | BlochMeasurement:
        """Return the measurement description as a Pauli measurement if possible, a Bloch measurement otherwise.

        Parameters
        ----------
        rel_tol : float, optional
            Relative tolerance for comparing angles, passed to :func:`math.isclose`.
            Default is ``1e-9``.
        abs_tol : float, optional
            Absolute tolerance for comparing angles, passed to :func:`math.isclose`.
            Default is ``0.0``.

        Returns
        -------
        PauliMeasurement | BlochMeasurement
            If ``self`` is already an instance of :class:`PauliMeasurement`, the function
            returns ``self``. If ``self`` is an instance of :class:`BlochMeasurement`, then
            either the measurement is close to a Pauli measurement (i.e., the angle is close to an
            integer multiple of π/2) and the corresponding Pauli measurement is returned,
            or it is not and ``self`` is returned.
        """

    @abstractmethod
    def subs(self, variable: Parameter, substitute: ExpressionOrSupportsFloat) -> Self:
        """Substitute a parameter with a value or expression in measurement angles."""

    @abstractmethod
    def xreplace(self, assignment: Mapping[Parameter, ExpressionOrSupportsFloat]) -> Self:
        """Perform parallel substitution of multiple parameters in measurement angles."""


@dataclass(frozen=True)
class BlochMeasurement(AbstractPlanarMeasurement, Measurement):
    r"""An MBQC measurement described by an angle and a plane.

    Attributes
    ----------
    angle : ExpressionOrFloat
        The angle of the measurement in units of :math:`\pi`. Should be between [0, 2).
    plane : graphix.fundamentals.Plane
        The measurement plane.
    """

    angle: ParameterizedAngle
    plane: Plane

    @override
    def __repr__(self) -> str:
        """Return an evaluable represention of the Bloch measurement.

        This representation assumes that :class:`Measurement` is in
        the scope.  The static methods ``Measurement.XY``,
        ``Measurement.YZ``, and ``Measurement.XZ`` are used to refer
        to the planes.

        """
        return f"Measurement.{self.plane.name}({self.angle})"

    @override
    def to_bloch(self) -> BlochMeasurement:
        """Return ``self`` (overridden from :class:`Measurement`)."""
        return self

    @override
    def downcast_bloch(self) -> BlochMeasurement:
        """Return ``self`` (overridden from :class:`Measurement`)."""
        return self

    @override
    def try_to_pauli(self, rel_tol: float = 1e-09, abs_tol: float = 0.0) -> PauliMeasurement | None:
        if not isinstance(self.angle, (int, float)):
            return None
        angle_double = 2 * self.angle
        angle_double_round = round(angle_double)
        if not math.isclose(angle_double, angle_double_round, rel_tol=rel_tol, abs_tol=abs_tol):
            return None
        angle_double_mod_4 = angle_double_round % 4
        axis = self.plane.cos if angle_double_mod_4 % 2 == 0 else self.plane.sin
        sign = Sign.minus_if(angle_double_mod_4 >= 2)
        return PauliMeasurement(axis, sign)

    @override
    def to_pauli_or_bloch(self, rel_tol: float = 1e-09, abs_tol: float = 0.0) -> PauliMeasurement | BlochMeasurement:
        pm = self.try_to_pauli(rel_tol=rel_tol, abs_tol=abs_tol)
        return self if pm is None else pm

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
        >>> Measurement.XY(0).isclose(Measurement.XY(0))
        True
        >>> Measurement.XY(0).isclose(Measurement.YZ(0))
        False
        >>> Measurement.XY(0.1).isclose(Measurement.XY(0))
        False
        """
        if not isinstance(other, Measurement):
            return False
        other_bloch = other.to_bloch()
        return (
            math.isclose(self.angle, other_bloch.angle, rel_tol=rel_tol, abs_tol=abs_tol)
            if isinstance(self.angle, float) and isinstance(other_bloch.angle, float)
            else self.angle == other_bloch.angle
        ) and self.plane == other_bloch.plane

    @override
    def to_plane(self) -> Plane:
        return self.plane

    @override
    def clifford(self, clifford_gate: Clifford) -> BlochMeasurement:
        new_plane = Plane.from_axes(*(PauliMeasurement(axis).clifford(clifford_gate).axis for axis in self.plane.axes))
        cos_pauli = PauliMeasurement(self.plane.cos).clifford(clifford_gate)
        sin_pauli = PauliMeasurement(self.plane.sin).clifford(clifford_gate)
        exchange = cos_pauli.axis != new_plane.cos
        angle = -self.angle if exchange == (cos_pauli.sign == sin_pauli.sign) else self.angle
        add_term: Angle = 0
        if cos_pauli.sign == Sign.MINUS:
            add_term += ANGLE_PI
        if exchange:
            add_term = ANGLE_PI / 2 - add_term
        angle += add_term
        return BlochMeasurement(angle, new_plane)

    @override
    def subs(self, variable: Parameter, substitute: ExpressionOrSupportsFloat) -> BlochMeasurement:
        return BlochMeasurement(parameter.subs(self.angle, variable, substitute), self.plane)

    @override
    def xreplace(self, assignment: Mapping[Parameter, ExpressionOrSupportsFloat]) -> BlochMeasurement:
        return BlochMeasurement(parameter.xreplace(self.angle, assignment), self.plane)


class PauliMeasurementMeta(ABCMeta):
    """Metaclass implementing `iter(PauliMeasurement)`."""

    def __iter__(cls) -> Iterator[PauliMeasurement]:
        """Iterate over Pauli measurements."""
        return (PauliMeasurement(axis, sign) for axis in Axis for sign in Sign)


@dataclass(frozen=True)
class PauliMeasurement(Measurement, metaclass=PauliMeasurementMeta):
    """Pauli measurement."""

    axis: Axis
    sign: Sign = Sign.PLUS

    @override
    def __repr__(self) -> str:
        """Return an evaluable represention of the Pauli measurement.

        This representation assumes that :class:`Measurement` is in
        the scope.  The class variables ``Measurement.X``,
        ``Measurement.Y``, and ``Measurement.Z`` are used to refer to
        the axes, and unary minus is used for negative sign.

        """
        result = f"Measurement.{self.axis.name}"
        if self.sign == Sign.MINUS:
            return f"-{result}"
        return result

    @override
    def __str__(self) -> str:
        return f"{self.sign}{self.axis.name}"

    def __pos__(self) -> PauliMeasurement:
        """Return the Pauli measurement itself.

        `+Measurement.X` is equivalent to `Measurement.X`.
        """
        return self

    def __neg__(self) -> PauliMeasurement:
        """Return the Pauli measurement with the opposite sign."""
        return PauliMeasurement(self.axis, -self.sign)

    def to_pauli(self) -> Pauli:
        """Return the Pauli gate.

        This method returns an instance of :class:`Pauli` and should
        not be confused with :meth:`try_to_pauli`, which overrides the
        method from :class:`Measurement`, and returns ``self``.

        """
        return self.sign * Pauli.from_axis(self.axis)

    @override
    def to_bloch(self) -> BlochMeasurement:
        match self.axis:
            case Axis.X:
                if self.sign == Sign.PLUS:
                    return Measurement.XY(0)
                return Measurement.XY(1)
            case Axis.Y:
                if self.sign == Sign.PLUS:
                    return Measurement.XY(0.5)
                return Measurement.XY(1.5)
            case Axis.Z:
                if self.sign == Sign.PLUS:
                    return Measurement.YZ(0)
                return Measurement.YZ(1)

    @override
    def downcast_bloch(self) -> BlochMeasurement:
        """Raise :class:`TypeError` (overridden from :class:`Measurement`)."""
        raise TypeError("Bloch measurement expected, but Pauli measurement was found.")

    @override
    def try_to_pauli(self, rel_tol: float = 1e-09, abs_tol: float = 0.0) -> PauliMeasurement:
        """Return ``self`` (overridden from :class:`Measurement`)."""
        return self

    @override
    def to_pauli_or_bloch(self, rel_tol: float = 1e-09, abs_tol: float = 0.0) -> PauliMeasurement:
        """Return ``self`` (overridden from :class:`Measurement`)."""
        return self

    @override
    def to_plane_or_axis(self) -> Axis:
        """Return ``self.axis`` (overridden from :class:`Measurement`)."""
        return self.axis

    @override
    def isclose(self, other: AbstractMeasurement, rel_tol: float = 1e-09, abs_tol: float = 0.0) -> bool:
        if isinstance(other, PauliMeasurement):
            return self == other
        return self.to_bloch().isclose(other, rel_tol, abs_tol)

    @override
    def clifford(self, clifford_gate: Clifford) -> PauliMeasurement:
        pauli = clifford_gate.measure(self.to_pauli())
        return PauliMeasurement(pauli.axis, pauli.unit.sign)

    @override
    def subs(self, variable: Parameter, substitute: ExpressionOrSupportsFloat) -> Self:
        return self

    @override
    def xreplace(self, assignment: Mapping[Parameter, ExpressionOrSupportsFloat]) -> Self:
        return self


# These fields have been declared in the definition of the
# ``Measurement`` class, but are only assigned here, now that
# ``PauliMeasurement`` is defined.
Measurement.X = PauliMeasurement(Axis.X)
Measurement.Y = PauliMeasurement(Axis.Y)
Measurement.Z = PauliMeasurement(Axis.Z)
