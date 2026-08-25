"""Instruction classes."""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, Literal, SupportsFloat, TypeAlias

# Self introduced in Python 3.11
# override introduced in Python 3.12
from typing_extensions import Self, override

from graphix import utils
from graphix.fundamentals import (
    Axis,
    ParameterizedAngle,
    Plane,
)
from graphix.pretty_print import OutputFormat, angle_to_str
from graphix.repr_mixins import DataclassReprMixin


def repr_angle(angle: ParameterizedAngle) -> str:
    """
    Return the representation string of an angle in radians.

    This is used for pretty-printing instructions with `angle` parameters.
    Delegates to :func:`pretty_print.angle_to_str`.
    """
    # Non-float-supporting objects are returned as-is
    if not isinstance(angle, SupportsFloat):
        return str(angle)

    return angle_to_str(angle, OutputFormat.ASCII)


class InstructionKind(Enum):
    """Tag for instruction kind."""

    CCX = enum.auto()
    RZZ = enum.auto()
    CNOT = enum.auto()
    SWAP = enum.auto()
    CZ = enum.auto()
    H = enum.auto()
    S = enum.auto()
    X = enum.auto()
    Y = enum.auto()
    Z = enum.auto()
    J = enum.auto()
    I = enum.auto()
    M = enum.auto()
    RX = enum.auto()
    RY = enum.auto()
    RZ = enum.auto()
    SDG = enum.auto()
    T = enum.auto()
    TDG = enum.auto()
    SX = enum.auto()
    SXDG = enum.auto()
    CY = enum.auto()
    P = enum.auto()
    U = enum.auto()
    CJ = enum.auto()
    CP = enum.auto()
    CRX = enum.auto()
    CRY = enum.auto()
    CRZ = enum.auto()
    CU = enum.auto()
    CSWAP = enum.auto()
    GPHASE = enum.auto()


class _KindChecker:
    """Enforce tag field declaration."""

    def __init_subclass__(cls) -> None:
        """Validate that subclasses define the ``kind`` attribute."""
        super().__init_subclass__()
        utils.check_kind(cls, {"InstructionKind": InstructionKind, "Plane": Plane})


class InstructionVisitor:
    """Visitor for instruction.

    This base class can be subclassed to rewrite instructions by
    overriding some of the following functions:

    - ``visit_qubit``: rewrite qubit indices.

    - ``visit_angle``: rewrite angles.

    - ``visit_axis``: rewrite axes.
    """

    def visit_qubit(self, qubit: int) -> int:
        """Rewrite a qubit index."""
        return qubit

    def visit_angle(self, angle: ParameterizedAngle) -> ParameterizedAngle:
        """Rewrite an angle."""
        return angle

    def visit_axis(self, axis: Axis) -> Axis:
        """Rewrite an axis."""
        return axis


class BaseInstruction(ABC, DataclassReprMixin):
    """Base class for circuit instructions."""

    @abstractmethod
    def visit(self, visitor: InstructionVisitor, *, copy: bool = False) -> Self:
        """Rewrite the instruction according to the given visitor.

        Parameters
        ----------
        visitor : InstructionVisitor
            The visitor specifying the rewriting.

        copy : bool, optional
            If ``True``, the current instruction remains unchanged, and a
            new instruction is returned. The default is ``False``, meaning
            that changes are performed in place.

        Returns
        -------
        Self
            The rewritten instruction. Equal to ``self`` if ``copy`` is ``False``.
        """


@dataclass(repr=False)
class CCX(_KindChecker, BaseInstruction):
    """Toffoli circuit instruction."""

    target: int
    controls: tuple[int, int]
    kind: ClassVar[Literal[InstructionKind.CCX]] = field(default=InstructionKind.CCX, init=False)

    @override
    def visit(self, visitor: InstructionVisitor, *, copy: bool = False) -> CCX:
        u, v = self.controls
        target = visitor.visit_qubit(self.target)
        controls = (visitor.visit_qubit(u), visitor.visit_qubit(v))
        if copy:
            return CCX(target, controls)
        self.target = target
        self.controls = controls
        return self


@dataclass(repr=False)
class RZZ(_KindChecker, BaseInstruction):
    """RZZ circuit instruction."""

    target: int
    control: int
    angle: ParameterizedAngle = field(metadata={"repr": repr_angle})
    kind: ClassVar[Literal[InstructionKind.RZZ]] = field(default=InstructionKind.RZZ, init=False)

    @override
    def visit(self, visitor: InstructionVisitor, *, copy: bool = False) -> RZZ:
        target = visitor.visit_qubit(self.target)
        control = visitor.visit_qubit(self.control)
        angle = visitor.visit_angle(self.angle)
        if copy:
            return RZZ(target, control, angle)
        self.target = target
        self.control = control
        self.angle = angle
        return self


@dataclass(repr=False)
class ControlledSingleTargetInstruction(BaseInstruction):
    """Base class for controlled single-target circuit instructions."""

    target: int
    control: int

    @override
    def visit(self, visitor: InstructionVisitor, *, copy: bool = False) -> Self:
        target = visitor.visit_qubit(self.target)
        control = visitor.visit_qubit(self.control)
        if copy:
            return type(self)(target, control)
        self.target = target
        self.control = control
        return self


@dataclass(repr=False)
class CY(_KindChecker, ControlledSingleTargetInstruction):
    """CY circuit instruction."""

    kind: ClassVar[Literal[InstructionKind.CY]] = field(default=InstructionKind.CY, init=False)


@dataclass(repr=False)
class CNOT(_KindChecker, ControlledSingleTargetInstruction):
    """CNOT circuit instruction."""

    kind: ClassVar[Literal[InstructionKind.CNOT]] = field(default=InstructionKind.CNOT, init=False)


# CZ is not defined as a ControlledSingleTargetInstruction because of
# the symmetry between the control and the target.
@dataclass(repr=False)
class CZ(_KindChecker, BaseInstruction):
    """CZ circuit instruction."""

    targets: tuple[int, int]
    kind: ClassVar[Literal[InstructionKind.CZ]] = field(default=InstructionKind.CZ, init=False)

    @override
    def visit(self, visitor: InstructionVisitor, *, copy: bool = False) -> CZ:
        u, v = self.targets
        targets = (visitor.visit_qubit(u), visitor.visit_qubit(v))
        if copy:
            return CZ(targets)
        self.targets = targets
        return self


@dataclass(repr=False)
class SWAP(_KindChecker, BaseInstruction):
    """SWAP circuit instruction."""

    targets: tuple[int, int]
    kind: ClassVar[Literal[InstructionKind.SWAP]] = field(default=InstructionKind.SWAP, init=False)

    @override
    def visit(self, visitor: InstructionVisitor, *, copy: bool = False) -> SWAP:
        u, v = self.targets
        targets = (visitor.visit_qubit(u), visitor.visit_qubit(v))
        if copy:
            return SWAP(targets)
        self.targets = targets
        return self


@dataclass(repr=False)
class CSWAP(_KindChecker, BaseInstruction):
    r"""CSWAP circuit instruction.

    The CSWAP gate applies the matrix

    .. math::

      \left[\begin{matrix}
        1 & 0 & 0 & 0\\
        0 & 1 & 0 & 0\\
        0 & 0 & \cos \frac \theta 2 & -\mathrm i \sin \frac \theta 2\\
        0 & 0 & -\mathrm i \sin \frac \theta 2 & \cos \frac \theta 2
      \end{matrix}\right]

    .. math::

      \left[\begin{matrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
      \end{matrix}\right]
    """

    control: int
    targets: tuple[int, int]
    kind: ClassVar[Literal[InstructionKind.CSWAP]] = field(default=InstructionKind.CSWAP, init=False)

    @override
    def visit(self, visitor: InstructionVisitor, *, copy: bool = False) -> CSWAP:
        control = visitor.visit_qubit(self.control)
        u, v = self.targets
        targets = (visitor.visit_qubit(u), visitor.visit_qubit(v))
        if copy:
            return CSWAP(control, targets)
        self.control = control
        self.targets = targets
        return self


@dataclass(repr=False)
class SingleTargetInstruction(BaseInstruction):
    """Base class for single-target circuit instructions."""

    target: int

    @override
    def visit(self, visitor: InstructionVisitor, *, copy: bool = False) -> Self:
        target = visitor.visit_qubit(self.target)
        if copy:
            return type(self)(target)
        self.target = target
        return self


@dataclass(repr=False)
class H(_KindChecker, SingleTargetInstruction):
    """H circuit instruction."""

    kind: ClassVar[Literal[InstructionKind.H]] = field(default=InstructionKind.H, init=False)


@dataclass(repr=False)
class S(_KindChecker, SingleTargetInstruction):
    """S circuit instruction."""

    kind: ClassVar[Literal[InstructionKind.S]] = field(default=InstructionKind.S, init=False)


@dataclass(repr=False)
class SDG(_KindChecker, SingleTargetInstruction):
    r"""SDG circuit instruction.

    The :math:`S^\dagger` gate applies the matrix
    :math:`\left[\begin{matrix}1 & 0\\0 & - \mathrm i\end{matrix}\right]`.

    We have :math:`S^\dagger = \mathrm e^{\mathrm i \frac \pi 4} R_Z(-\frac \pi 2)`.
    """

    kind: ClassVar[Literal[InstructionKind.SDG]] = field(default=InstructionKind.SDG, init=False)


@dataclass(repr=False)
class T(_KindChecker, SingleTargetInstruction):
    r"""T circuit instruction.

    The :math:`T` gate applies the matrix
    :math:`\left[\begin{matrix}1 & 0\\0 & \mathrm e^{\mathrm i \frac \pi 4}\end{matrix}\right]`.

    We have :math:`T = \mathrm e^{\mathrm i \frac \pi 8} R_Z(\frac \pi 4)`.
    """

    kind: ClassVar[Literal[InstructionKind.T]] = field(default=InstructionKind.T, init=False)


@dataclass(repr=False)
class TDG(_KindChecker, SingleTargetInstruction):
    r"""TDG circuit instruction.

    The :math:`T^\dagger` gate applies the matrix
    :math:`\left[\begin{matrix}1 & 0\\0 & \mathrm e^{- \mathrm i \frac \pi 4}\end{matrix}\right]`.

    We have :math:`T^\dagger = \mathrm e^{\mathrm i \frac \pi 8} R_Z(- \frac \pi 4)`.
    """

    kind: ClassVar[Literal[InstructionKind.TDG]] = field(default=InstructionKind.TDG, init=False)


@dataclass(repr=False)
class SX(_KindChecker, SingleTargetInstruction):
    r"""SX circuit instruction.

    The :math:`SX` (:math:`\sqrt X`) gate applies the matrix
    :math:`\frac 1 2 \left[\begin{matrix}1 + \mathrm i & 1 - \mathrm i\\1 - \mathrm i & 1 + \mathrm i\end{matrix}\right]`.

    We have :math:`SX = \mathrm e^{\mathrm i \frac \pi 4} R_X(\frac \pi 2)`.
    """

    kind: ClassVar[Literal[InstructionKind.SX]] = field(default=InstructionKind.SX, init=False)


@dataclass(repr=False)
class SXDG(_KindChecker, SingleTargetInstruction):
    r"""SXDG circuit instruction.

    The :math:`SX^\dagger` (:math:`{\sqrt X}^\dagger`) gate applies the matrix
    :math:`\frac 1 2 \left[\begin{matrix}1 - \mathrm i & 1 + \mathrm i\\1 + \mathrm i & 1 - \mathrm i\end{matrix}\right]`.

    We have :math:`SX^\dagger = \mathrm e^{\mathrm i \frac \pi 4} R_X(-\frac \pi 2)`.
    """

    kind: ClassVar[Literal[InstructionKind.SXDG]] = field(default=InstructionKind.SXDG, init=False)


@dataclass(repr=False)
class X(_KindChecker, SingleTargetInstruction):
    """X circuit instruction."""

    kind: ClassVar[Literal[InstructionKind.X]] = field(default=InstructionKind.X, init=False)


@dataclass(repr=False)
class Y(_KindChecker, SingleTargetInstruction):
    """Y circuit instruction."""

    kind: ClassVar[Literal[InstructionKind.Y]] = field(default=InstructionKind.Y, init=False)


@dataclass(repr=False)
class Z(_KindChecker, SingleTargetInstruction):
    """Z circuit instruction."""

    kind: ClassVar[Literal[InstructionKind.Z]] = field(default=InstructionKind.Z, init=False)


@dataclass(repr=False)
class I(_KindChecker, SingleTargetInstruction):
    """I circuit instruction."""

    kind: ClassVar[Literal[InstructionKind.I]] = field(default=InstructionKind.I, init=False)


@dataclass(repr=False)
class M(_KindChecker, BaseInstruction):
    """M circuit instruction."""

    target: int
    axis: Axis
    kind: ClassVar[Literal[InstructionKind.M]] = field(default=InstructionKind.M, init=False)

    @override
    def visit(self, visitor: InstructionVisitor, *, copy: bool = False) -> M:
        target = visitor.visit_qubit(self.target)
        axis = visitor.visit_axis(self.axis)
        if copy:
            return M(target, axis)
        self.target = target
        self.axis = axis
        return self


@dataclass(repr=False)
class RotationInstruction(BaseInstruction):
    """Base class for rotation instructions."""

    target: int
    angle: ParameterizedAngle = field(metadata={"repr": repr_angle})

    @override
    def visit(self, visitor: InstructionVisitor, *, copy: bool = False) -> Self:
        target = visitor.visit_qubit(self.target)
        angle = visitor.visit_angle(self.angle)
        if copy:
            return type(self)(target, angle)
        self.target = target
        self.angle = angle
        return self


@dataclass(repr=False)
class P(_KindChecker, RotationInstruction):
    r"""P rotation circuit instruction.

    The :math:`P(\theta)` gate applies the matrix

    .. math::

      \left[\begin{matrix}
        1 & 0\\
        0 & \mathrm e^{\mathrm i \theta}
      \end{matrix}\right]

    We have :math:`P(\theta) = \mathrm e^{\theta/2} R_Z(\theta)`.
    """

    kind: ClassVar[Literal[InstructionKind.P]] = field(default=InstructionKind.P, init=False)


@dataclass(repr=False)
class RX(_KindChecker, RotationInstruction):
    """X rotation circuit instruction."""

    kind: ClassVar[Literal[InstructionKind.RX]] = field(default=InstructionKind.RX, init=False)


@dataclass(repr=False)
class RY(_KindChecker, RotationInstruction):
    """Y rotation circuit instruction."""

    kind: ClassVar[Literal[InstructionKind.RY]] = field(default=InstructionKind.RY, init=False)


@dataclass(repr=False)
class RZ(_KindChecker, RotationInstruction):
    """Z rotation circuit instruction."""

    kind: ClassVar[Literal[InstructionKind.RZ]] = field(default=InstructionKind.RZ, init=False)


@dataclass(repr=False)
class J(_KindChecker, RotationInstruction):
    """J circuit instruction."""

    kind: ClassVar[Literal[InstructionKind.J]] = field(default=InstructionKind.J, init=False)


@dataclass(repr=False)
class U(_KindChecker, BaseInstruction):
    r"""U circuit instruction.

    The :math:`U(\theta, \phi, \lambda)` gate applies the matrix

    .. math::

      \left[\begin{matrix}
        \cos \frac \theta 2 & - \mathrm e^{\mathrm i\lambda} \sin \frac \theta 2 \\
        \mathrm e^{\mathrm i \phi} \sin \frac \theta 2 & \mathrm e^{\mathrm i (\phi + \lambda)} \cos \frac \theta 2
      \end{matrix}\right]

    It can be decomposed as

    .. math::

      U(\theta, \phi, \lambda) = \mathrm e^{\mathrm i\frac{\phi + \lambda}{2}} R_Z(\phi) R_Y(\theta) R_Z(\lambda)
      = \mathrm e^{\mathrm i\frac{\theta}{2}}
        H J\left(\phi + \frac{\pi}{2}\right)
        J(\theta)
        J\left(\lambda - \frac{\pi}{2}\right)
    """

    target: int
    theta: ParameterizedAngle = field(metadata={"repr": repr_angle})
    phi: ParameterizedAngle = field(metadata={"repr": repr_angle})
    lambda_: ParameterizedAngle = field(metadata={"repr": repr_angle})
    kind: ClassVar[Literal[InstructionKind.U]] = field(default=InstructionKind.U, init=False)

    @override
    def visit(self, visitor: InstructionVisitor, *, copy: bool = False) -> Self:
        target = visitor.visit_qubit(self.target)
        theta = visitor.visit_angle(self.theta)
        phi = visitor.visit_angle(self.phi)
        lambda_ = visitor.visit_angle(self.lambda_)
        if copy:
            return type(self)(target, theta, phi, lambda_)
        self.target = target
        self.theta = theta
        self.phi = phi
        self.lambda_ = lambda_
        return self


@dataclass(repr=False)
class CU(_KindChecker, BaseInstruction):
    r"""Controlled-U circuit instruction.

    The :math:`CU(\theta, \phi, \lambda, \gamma)` gate applies the matrix

    .. math::

      \left[\begin{matrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & \mathrm e^{\mathrm i\gamma}
          \cos\left(\frac{\theta}{2}\right) &
          -\mathrm e^{\mathrm i(\gamma + \lambda)}
          \sin\left(\frac{\theta}{2}\right) \\
        0 & 0 & \mathrm e^{\mathrm i(\gamma + \phi)}
          \sin\left(\frac{\theta}{2}\right) &
          \mathrm e^{\mathrm i(\gamma + \phi + \lambda)}
          \cos\left(\frac{\theta}{2}\right)
      \end{matrix}\right]

    It can be decomposed as

    .. math::

      CU(\theta, \phi, \lambda, \gamma) =
        \left(P\left(\frac{\gamma - \theta} 2\right) \otimes I)
        CJ(0) CJ\left(\phi + \frac \pi 2\right)
        CJ(\theta) CJ\left(\lambda - \frac \pi 2\right)
    """

    control: int
    target: int
    theta: ParameterizedAngle = field(metadata={"repr": repr_angle})
    phi: ParameterizedAngle = field(metadata={"repr": repr_angle})
    lambda_: ParameterizedAngle = field(metadata={"repr": repr_angle})
    gamma: ParameterizedAngle = field(metadata={"repr": repr_angle})
    kind: ClassVar[Literal[InstructionKind.CU]] = field(default=InstructionKind.CU, init=False)

    @override
    def visit(self, visitor: InstructionVisitor, *, copy: bool = False) -> Self:
        control = visitor.visit_qubit(self.control)
        target = visitor.visit_qubit(self.target)
        theta = visitor.visit_angle(self.theta)
        phi = visitor.visit_angle(self.phi)
        lambda_ = visitor.visit_angle(self.lambda_)
        gamma = visitor.visit_angle(self.gamma)
        if copy:
            return type(self)(control, target, theta, phi, lambda_, gamma)
        self.control = control
        self.target = target
        self.theta = theta
        self.phi = phi
        self.lambda_ = lambda_
        self.gamma = gamma
        return self


@dataclass(repr=False)
class ControlledRotationInstruction(BaseInstruction):
    """Base class for rotation instructions."""

    target: int
    control: int
    angle: ParameterizedAngle = field(metadata={"repr": repr_angle})

    @override
    def visit(self, visitor: InstructionVisitor, *, copy: bool = False) -> Self:
        target = visitor.visit_qubit(self.target)
        control = visitor.visit_qubit(self.control)
        angle = visitor.visit_angle(self.angle)
        if copy:
            return type(self)(target, control, angle)
        self.target = target
        self.control = control
        self.angle = angle
        return self


@dataclass(repr=False)
class CP(_KindChecker, ControlledRotationInstruction):
    r"""Controlled-P rotation circuit instruction.

    The :math:`CP(\theta)` gate applies the matrix

    .. math::

      \left[\begin{matrix}
        1 & 0 & 0 & 0\\
        0 & 1 & 0 & 0\\
        0 & 0 & 1 & 0\\
        0 & 0 & 0 & \mathrm e^{\mathrm i \theta}
      \end{matrix}\right]
    """

    kind: ClassVar[Literal[InstructionKind.CP]] = field(default=InstructionKind.CP, init=False)


@dataclass(repr=False)
class CRX(_KindChecker, ControlledRotationInstruction):
    r"""Controlled-X rotation circuit instruction.

    The :math:`CRX(\theta)` gate applies the matrix

    .. math::

      \left[\begin{matrix}
        1 & 0 & 0 & 0\\
        0 & 1 & 0 & 0\\
        0 & 0 & \cos \frac \theta 2 & -\mathrm i \sin \frac \theta 2\\
        0 & 0 & -\mathrm i \sin \frac \theta 2 & \cos \frac \theta 2
      \end{matrix}\right]
    """

    kind: ClassVar[Literal[InstructionKind.CRX]] = field(default=InstructionKind.CRX, init=False)


@dataclass(repr=False)
class CRY(_KindChecker, ControlledRotationInstruction):
    r"""Controlled-Y rotation circuit instruction.

    .. math::

      \left[\begin{matrix}
        1 & 0 & 0 & 0\\
        0 & 1 & 0 & 0\\
        0 & 0 & \cos \frac \theta 2 & - \sin \frac \theta 2\\
        0 & 0 & \sin \frac \theta 2 & \cos \frac \theta 2
      \end{matrix}\right]
    """

    kind: ClassVar[Literal[InstructionKind.CRY]] = field(default=InstructionKind.CRY, init=False)


@dataclass(repr=False)
class CRZ(_KindChecker, ControlledRotationInstruction):
    r"""Controlled-Z rotation circuit instruction.

    .. math::

      \left[\begin{matrix}
        1 & 0 & 0 & 0\\
        0 & 1 & 0 & 0\\
        0 & 0 & \mathrm e^{-\mathrm i \frac \theta 2} & 0\\
        0 & 0 & 0 & \mathrm e^{\mathrm i \frac \theta 2}
      \end{matrix}\right]
    """

    kind: ClassVar[Literal[InstructionKind.CRZ]] = field(default=InstructionKind.CRZ, init=False)


@dataclass(repr=False)
class CJ(_KindChecker, ControlledRotationInstruction):
    r"""Controlled-J circuit instruction.

    The :math:`CJ(\alpha)` gate applies the matrix

    .. math::

      \left[\begin{matrix}
        1 & 0 & 0 & 0\\
        0 & 1 & 0 & 0\\
        0 & 0 & \frac 1 {\sqrt 2} & \frac 1 {\sqrt 2} \mathrm e^{\mathrm i \alpha}\\
        0 & 0 & \frac 1 {\sqrt 2} & - \frac 1 {\sqrt 2} \mathrm e^{\mathrm i \alpha}
      \end{matrix}\right]

    Following Lemmas 4.3 and 5.1 of Barenco et al. (1995), we define:

    .. math::

      \begin{aligned}
        A &= R_Y\left(\frac \pi 4\right),\\
        B &= R_Y\left(- \frac \pi 4\right) R_Z(- \delta),\\
        C &= R_Z(\delta),\\
        \delta &= \frac {\alpha + \pi} 2
      \end{aligned}

    These operators satisfy :math:`ABC = I` and
    :math:`AXBXC = \mathrm e^{-\mathrm i \delta} J(\alpha)` with
    :math:``.

    Consequently, :math:`CJ(\alpha)` can be decomposed as:

    .. math::

      CJ(\alpha) = (P(\delta) \otimes I) \, (I \otimes A) \, CX \, (I \otimes B) \, CX \, (I \otimes C)

    References
    ----------
    Barenco, A., Bennett, C. H., Cleve, R., DiVincenzo, D. P., Margolus, N., Shor, P., Sleator, T., Smolin, J. A., & Weinfurter, H. (1995).
    Elementary gates for quantum computation. Physical Review A, 52(5), 3457-3467.
    https://doi.org/10.1103/physreva.52.3457
    """

    kind: ClassVar[Literal[InstructionKind.CJ]] = field(default=InstructionKind.CJ, init=False)


@dataclass(repr=False)
class GPHASE(_KindChecker, BaseInstruction):
    """GPHASE circuit instruction."""

    angle: ParameterizedAngle = field(metadata={"repr": repr_angle})
    kind: ClassVar[Literal[InstructionKind.GPHASE]] = field(default=InstructionKind.GPHASE, init=False)

    @override
    def visit(self, visitor: InstructionVisitor, *, copy: bool = False) -> GPHASE:
        angle = visitor.visit_angle(self.angle)
        if copy:
            return GPHASE(angle)
        self.angle = angle
        return self


class Instruction:
    """Grouping of all instructions for namespace exposure.

    Notes
    -----
    This class is not meant to be instantiated, but rather serves as a namespace for all instructions except RZZ.
    The type alias for "any command" is :data:`InstructionKind`.
    """

    I: TypeAlias = I
    X: TypeAlias = X
    Y: TypeAlias = Y
    Z: TypeAlias = Z
    H: TypeAlias = H
    S: TypeAlias = S
    SDG: TypeAlias = SDG
    T: TypeAlias = T
    TDG: TypeAlias = TDG
    SX: TypeAlias = SX
    SXDG: TypeAlias = SXDG
    J: TypeAlias = J
    P: TypeAlias = P
    RX: TypeAlias = RX
    RY: TypeAlias = RY
    RZ: TypeAlias = RZ
    U: TypeAlias = U
    CJ: TypeAlias = CJ
    CP: TypeAlias = CP
    CRX: TypeAlias = CRX
    CRY: TypeAlias = CRY
    CRZ: TypeAlias = CRZ
    CU: TypeAlias = CU
    CNOT: TypeAlias = CNOT
    CY: TypeAlias = CY
    CZ: TypeAlias = CZ
    CCX: TypeAlias = CCX
    RZZ: TypeAlias = RZZ
    SWAP: TypeAlias = SWAP
    CSWAP: TypeAlias = CSWAP
    M: TypeAlias = M
    GPHASE: TypeAlias = GPHASE

    def __init__(self) -> None:
        raise TypeError("Instruction is a namespace, not a class.")


if TYPE_CHECKING:
    InstructionType = (
        I
        | X
        | Y
        | Z
        | H
        | S
        | SDG
        | T
        | TDG
        | SX
        | SXDG
        | J
        | P
        | RX
        | RY
        | RZ
        | U
        | CJ
        | CP
        | CRX
        | CRY
        | CRZ
        | CU
        | CNOT
        | CY
        | CZ
        | CCX
        | RZZ
        | SWAP
        | CSWAP
        | M
        | GPHASE
    )
