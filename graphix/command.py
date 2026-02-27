"""Data validator command classes."""

from __future__ import annotations

import dataclasses
import enum
import logging
from enum import Enum
from typing import ClassVar, Generic, Literal, TypeVar, cast

from graphix import utils
from graphix.clifford import Clifford, Domains
from graphix.fundamentals import Angle, ParameterizedAngle
from graphix.measurements import Measurement
from graphix.repr_mixins import DataclassReprMixin
from graphix.states import BasicStates, State

Node = int

logger = logging.getLogger(__name__)

AngleT = TypeVar("AngleT", ParameterizedAngle, Angle)

AngleT_co = TypeVar("AngleT_co", ParameterizedAngle, Angle, covariant=True)


class CommandKind(Enum):
    """Tag for command kind."""

    N = enum.auto()
    M = enum.auto()
    E = enum.auto()
    C = enum.auto()
    X = enum.auto()
    Z = enum.auto()
    S = enum.auto()
    T = enum.auto()
    ApplyNoise = enum.auto()  # see noise_models/noise_model.py


class _KindChecker:
    """Enforce tag field declaration."""

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        utils.check_kind(cls, {"CommandKind": CommandKind, "Clifford": Clifford})


class BaseCommand(DataclassReprMixin):
    """Base class for pattern command."""


@dataclasses.dataclass(repr=False)
class BaseN(BaseCommand):
    r"""Base preparation command.

    Represent a preparation of a node. In `graphix`, a preparation is
    an instance of class `N`, with an initial state (defaults to
    :class:`~graphix.states.BasicStates.PLUS`). The base class `BaseN`
    allows users to define new class of preparation commands with
    different abstractions.  For example, in the context of blind
    computations, the server only knows which node is prepared, and
    the initial state are given by the
    :class:`graphix.simulator.PrepareMethod` provided by the client.

    Parameters
    ----------
    node : int
        Index of the qubit to prepare.
    """

    node: int
    kind: ClassVar[Literal[CommandKind.N]] = dataclasses.field(default=CommandKind.N, init=False)


@dataclasses.dataclass(repr=False)
class N(BaseN, _KindChecker):
    r"""Preparation command.

    Parameters
    ----------
    node : int
        Index of the qubit to prepare.
    state : ~graphix.states.State, optional
        Initial state, defaults to :class:`~graphix.states.BasicStates.PLUS`.
    """

    state: State = dataclasses.field(default_factory=lambda: BasicStates.PLUS)
    kind: ClassVar[Literal[CommandKind.N]] = dataclasses.field(default=CommandKind.N, init=False)


@dataclasses.dataclass(repr=False)
class BaseM(BaseCommand):
    """Base measurement command.

    Represent a measurement of a node. In `graphix`, a measurement is an instance of
    class `M`, with given plane, angles, and domains. The base class `BaseM` allows users to define
    new class of measurements with different abstractions. For example, in the context
    of blind computations, the server only knows which node is measured, and the parameters
    are given by the :class:`graphix.simulator.MeasureMethod` provided by the client.
    """

    node: Node
    kind: ClassVar[Literal[CommandKind.M]] = dataclasses.field(default=CommandKind.M, init=False)


@dataclasses.dataclass(repr=False)
class M(BaseM, _KindChecker, Generic[AngleT_co]):
    r"""Measurement command.

    Parameters
    ----------
    node : int
        Node index of the measured qubit.
    measurement : Measurement
        Measurement description.
    s_domain : set[int], optional
        Domain for the X byproduct operator.
    t_domain : set[int], optional
        Domain for the Z byproduct operator.
    """

    measurement: Measurement[AngleT_co] = cast("Measurement[AngleT_co]", Measurement.X)  # noqa: RUF009
    s_domain: set[Node] = dataclasses.field(default_factory=set)
    t_domain: set[Node] = dataclasses.field(default_factory=set)
    kind: ClassVar[Literal[CommandKind.M]] = dataclasses.field(default=CommandKind.M, init=False)

    def clifford(self, clifford_gate: Clifford) -> M[AngleT_co]:
        r"""Return a new measurement command with a Clifford applied.

        Parameters
        ----------
        clifford_gate : ~graphix.clifford.Clifford
            Clifford gate to apply before the measurement.

        Returns
        -------
        :class:`M`
            Equivalent command representing the pattern ``MC``.
        """
        domains = clifford_gate.commute_domains(Domains(self.s_domain, self.t_domain))
        return M(
            self.node,
            self.measurement.clifford(clifford_gate),
            domains.s_domain,
            domains.t_domain,
        )


@dataclasses.dataclass(repr=False)
class E(_KindChecker, BaseCommand):
    r"""Entanglement command between two qubits.

    Parameters
    ----------
    nodes : tuple[int, int]
        Pair of nodes to entangle.
    """

    nodes: tuple[Node, Node]
    kind: ClassVar[Literal[CommandKind.E]] = dataclasses.field(default=CommandKind.E, init=False)


@dataclasses.dataclass(repr=False)
class C(_KindChecker, BaseCommand):
    r"""Local Clifford gate command.

    Parameters
    ----------
    node : int
        Node index on which to apply the gate.
    clifford : ~graphix.clifford.Clifford
        Clifford operator to apply.
    """

    node: Node
    clifford: Clifford
    kind: ClassVar[Literal[CommandKind.C]] = dataclasses.field(default=CommandKind.C, init=False)


@dataclasses.dataclass(repr=False)
class X(_KindChecker, BaseCommand):
    r"""X correction command.

    Parameters
    ----------
    node : int
        Node to correct.
    domain : set[int], optional
        Domain for the byproduct operator.
    """

    node: Node
    domain: set[Node] = dataclasses.field(default_factory=set)
    kind: ClassVar[Literal[CommandKind.X]] = dataclasses.field(default=CommandKind.X, init=False)


@dataclasses.dataclass(repr=False)
class Z(_KindChecker, BaseCommand):
    r"""Z correction command.

    Parameters
    ----------
    node : int
        Node to correct.
    domain : set[int], optional
        Domain for the byproduct operator.
    """

    node: Node
    domain: set[Node] = dataclasses.field(default_factory=set)
    kind: ClassVar[Literal[CommandKind.Z]] = dataclasses.field(default=CommandKind.Z, init=False)


@dataclasses.dataclass(repr=False)
class S(_KindChecker, BaseCommand):
    r"""S command.

    Parameters
    ----------
    node : int
        Node for the byproduct operator.
    domain : set[int], optional
        Domain on which to apply the operator.
    """

    node: Node
    domain: set[Node] = dataclasses.field(default_factory=set)
    kind: ClassVar[Literal[CommandKind.S]] = dataclasses.field(default=CommandKind.S, init=False)


@dataclasses.dataclass(repr=False)
class T(_KindChecker, BaseCommand):
    r"""T command.

    Parameters
    ----------
    None
        The T command acts globally without parameters.
    """

    kind: ClassVar[Literal[CommandKind.T]] = dataclasses.field(default=CommandKind.T, init=False)


Command = N | M[AngleT_co] | E | C | X | Z | S | T
Correction = X | Z
