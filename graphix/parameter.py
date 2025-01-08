"""Parameter class

Parameter object acts as a placeholder of measurement angles and
allows the manipulation of the measurement pattern without specific
value assignment.

"""

from __future__ import annotations

import sys
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, SupportsComplex, SupportsFloat, TypeVar


class Expression(ABC):
    """Expression with parameters."""

    @abstractmethod
    def __mul__(self, other: Any) -> ExpressionOrFloat: ...

    @abstractmethod
    def __rmul__(self, other: Any) -> ExpressionOrFloat: ...

    @abstractmethod
    def __add__(self, other: Any) -> ExpressionOrFloat: ...

    @abstractmethod
    def __radd__(self, other: Any) -> ExpressionOrFloat: ...

    @abstractmethod
    def __sub__(self, other: Any) -> ExpressionOrFloat: ...

    @abstractmethod
    def __rsub__(self, other: Any) -> ExpressionOrFloat: ...

    @abstractmethod
    def __neg__(self) -> ExpressionOrFloat: ...

    @abstractmethod
    def __truediv__(self, other: Any) -> ExpressionOrFloat: ...

    @abstractmethod
    def __rtruediv__(self, other: Any) -> ExpressionOrFloat: ...

    @abstractmethod
    def __mod__(self, other: Any) -> ExpressionOrFloat: ...

    @abstractmethod
    def subs(self, variable: Parameter, value: ExpressionOrFloat) -> ExpressionOrComplex: ...

    @abstractmethod
    def xreplace(self, assignment: Mapping[Parameter, ExpressionOrFloat]) -> ExpressionOrComplex: ...


class Parameter(Expression):
    """Abstract class for substituable parameter."""

    ...


class PlaceholderOperationError(ValueError):
    def __init__(self):
        super().__init__(
            "Placeholder angles do not support any form of computation before substitution except affine operation. You may use `subs` with an actual value before the computation."
        )


@dataclass
class AffineExpression(Expression):
    """Affine expression.

    An affine expression is of the form `a*x+b` where `a` and `b` are numbers and `x` is a parameter.
    """

    a: float
    x: Parameter
    b: float

    def offset(self, d: float) -> AffineExpression:
        return AffineExpression(a=self.a, x=self.x, b=self.b + d)

    def __scale_non_null(self, k: float) -> AffineExpression:
        return AffineExpression(a=k * self.a, x=self.x, b=k * self.b)

    def scale(self, k: float) -> ExpressionOrFloat:
        if k == 0:
            return 0
        return self.__scale_non_null(k)

    def __mul__(self, other: Any) -> ExpressionOrFloat:
        if isinstance(other, SupportsFloat):
            return self.scale(float(other))
        return NotImplemented

    def __rmul__(self, other: Any) -> ExpressionOrFloat:
        if isinstance(other, SupportsFloat):
            return self.scale(float(other))
        return NotImplemented

    def __add__(self, other: Any) -> ExpressionOrFloat:
        if isinstance(other, SupportsFloat):
            return self.offset(float(other))
        if isinstance(other, AffineExpression):
            if other.x != self.x:
                raise PlaceholderOperationError()
            a = self.a + other.a
            if a == 0:
                return 0
            return AffineExpression(a=a, x=self.x, b=self.b + other.b)
        return NotImplemented

    def __radd__(self, other: Any) -> ExpressionOrFloat:
        if isinstance(other, SupportsFloat):
            return self.offset(float(other))
        return NotImplemented

    def __sub__(self, other: Any) -> ExpressionOrFloat:
        if isinstance(other, AffineExpression):
            return self + -other
        if isinstance(other, SupportsFloat):
            return self + -float(other)
        return NotImplemented

    def __rsub__(self, other: Any) -> ExpressionOrFloat:
        if isinstance(other, SupportsFloat):
            return self.__scale_non_null(-1).offset(float(other))
        return NotImplemented

    def __neg__(self) -> ExpressionOrFloat:
        return self.__scale_non_null(-1)

    def __truediv__(self, other: Any) -> ExpressionOrFloat:
        if isinstance(other, SupportsFloat):
            return self.scale(1 / float(other))
        return NotImplemented

    def __rtruediv__(self, other: Any) -> ExpressionOrFloat:
        return NotImplemented

    def __mod__(self, other: Any) -> ExpressionOrFloat:
        return NotImplemented

    def __str__(self) -> str:
        return f"{self.a} * {self.x} + {self.b}"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, AffineExpression):
            return self.a == other.a and self.x == other.x and self.b == other.b
        return False

    def evaluate(self, value: ExpressionOrSupportsFloat) -> ExpressionOrFloat:
        return self.a * float(value) + self.b

    def subs(self, variable: Parameter, value: ExpressionOrSupportsFloat) -> ExpressionOrComplex:
        if variable == self.x:
            return self.evaluate(value)
        return self

    def xreplace(self, assignment: Mapping[Parameter, ExpressionOrSupportsFloat]) -> ExpressionOrComplex:
        value = assignment.get(self.x)
        # `value` can be 0, so checking with `is not None` is mandatory here.
        if value is not None:
            return self.evaluate(value)
        return self


class Placeholder(AffineExpression, Parameter):
    """Placeholder for measurement angles.

    These placeholder may appear in affine expressions.  Placeholders
    and affine expressions may be used as angles in rotation gates of
    :class:`Circuit` class or for the measurement angle of the
    measurement commands.  Pattern optimizations such that
    standardization, signal shifting and Pauli preprocessing can be
    applied to patterns with placeholders.

    These placeholders and affine expressions do not support arbitrary
    computation and are not suitable for simulation.  You may use
    :func:`Circuit.subs` or :func:`Pattern.subs` with an actual value
    before the computation.

    """

    def __init__(self, name: str) -> None:
        """Create a new :class:`Placeholder` object.

        Parameters
        ----------
        name : str
            name of the parameter, used for binding values.
        """
        self.__name = name
        super().__init__(a=1, x=self, b=0)

    @property
    def name(self) -> str:
        return self.__name

    def __repr__(self) -> str:
        return f"Placeholder({self.__name!r})"

    def __str__(self) -> str:
        return self.__name

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Parameter):
            return self is other
        return super().__eq__(other)

    def __hash__(self) -> int:
        return id(self)


if sys.version_info >= (3, 10):
    ExpressionOrFloat = Expression | float

    ExpressionOrComplex = Expression | complex

    ExpressionOrSupportsFloat = Expression | SupportsFloat

    ExpressionOrSupportsComplex = Expression | SupportsComplex
else:
    ExpressionOrFloat = typing.Union[Expression, float]

    ExpressionOrComplex = typing.Union[Expression, complex]

    ExpressionOrSupportsFloat = typing.Union[Expression, SupportsFloat]

    ExpressionOrSupportsComplex = typing.Union[Expression, SupportsComplex]


T = TypeVar("T")


def subs(value: T, variable: Parameter, substitute: ExpressionOrSupportsFloat) -> T | complex:
    """Generic substitution in `value`: if `value` is in instance of
    :class:`Expression`, then return `value.subs(variable,
    substitute)` (coerced into a complex if the result is a number).
    If `value` does not implement `subs`, `value` is returned
    unchanged.

    This function is used to apply substitution to collections where
    some elements are `Expression` and other elements are just
    plain numbers.

    """
    if not isinstance(value, Expression):
        print(f"{value} is not an expression")
        return value
    new_value = value.subs(variable, substitute)
    if isinstance(new_value, SupportsComplex):
        return complex(new_value)
    return new_value


def xreplace(value: T, assignment: Mapping[Parameter, ExpressionOrSupportsFloat]) -> T | complex:
    """Generic parallel substitution in `value`: if `value` is an an
    instance of :class:`Expression`, then return
    `value.xreplace(assignment)` (coerced into a complex if the result
    is a number).  If `value` does not implement `xreplace`, `value`
    is returned unchanged.

    This function is used to apply parallel substitutions to
    collections where some elements are Expression and other elements
    are just plain numbers.

    """
    if not isinstance(value, Expression):
        return value
    new_value = value.xreplace(assignment)
    if isinstance(new_value, SupportsComplex):
        return complex(new_value)
    return new_value
