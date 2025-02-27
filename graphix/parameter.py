"""Parameter class.

Parameter object acts as a placeholder of measurement angles and
allows the manipulation of the measurement pattern without specific
value assignment.

"""

from __future__ import annotations

import math
import sys
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, SupportsComplex, SupportsFloat, TypeVar, overload

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping


class Expression(ABC):
    """Expression with parameters."""

    @abstractmethod
    def __mul__(self, other: object) -> ExpressionOrFloat:
        """
        Return the product of this expression with another object.

        This special method is called to implement the multiplication operator (*).
        """

    @abstractmethod
    def __rmul__(self, other: object) -> ExpressionOrFloat:
        """
        Return the product of `other` with this expression.

        This special method is called to implement the multiplication operator (*)
        when the left operand does not support multiplication with this type.
        Typically, `other` can be a number.
        """

    @abstractmethod
    def __add__(self, other: object) -> ExpressionOrFloat:
        """
        Return the sum of this expression with another object.

        This special method is called to implement the addition operator (+).
        """

    @abstractmethod
    def __radd__(self, other: object) -> ExpressionOrFloat:
        """
        Return the sum of `other` with this expression.

        This special method is called to implement the addition operator (+)
        when the left operand does not support addition with this type.
        Typically, `other` can be a number.
        """

    @abstractmethod
    def __sub__(self, other: object) -> ExpressionOrFloat:
        """
        Return the difference of this expression with another object.

        This special method is called to implement the substraction operator (-).
        """

    @abstractmethod
    def __rsub__(self, other: object) -> ExpressionOrFloat:
        """
        Return the difference of `other` with this expression.

        This special method is called to implement the substraction operator (-)
        when the left operand does not support substraction with this type.
        Typically, `other` can be a number.
        """

    @abstractmethod
    def __neg__(self) -> ExpressionOrFloat:
        """
        Return the opposite of this expression.

        This special method is called to implement the unary opposite operator (-).
        """

    @abstractmethod
    def __truediv__(self, other: object) -> ExpressionOrFloat:
        """
        Return the quotient of this expression with another object.

        This special method is called to implement the division operator (/).
        """

    @abstractmethod
    def subs(self, variable: Parameter, value: ExpressionOrSupportsFloat) -> ExpressionOrComplex:
        """Return the expression where every occurrence of `variable` is replaced with `value`."""

    @abstractmethod
    def xreplace(self, assignment: Mapping[Parameter, ExpressionOrSupportsFloat]) -> ExpressionOrComplex:
        """
        Return the expression where every occurrence of any keys from `assignment` is replaced with the corresponding value.

        The substitutions are performed in parallel, i.e., once an
        occurrence has been replaced by a value, this value is not
        subject to any further replacement, even if another occurrence
        of a key appears in this value.
        """


class ExpressionWithTrigonometry(Expression, ABC):
    """Expression that supports trigonometric functions."""

    @abstractmethod
    def cos(self) -> ExpressionWithTrigonometry:
        """Return the cosine of the expression."""

    @abstractmethod
    def sin(self) -> ExpressionWithTrigonometry:
        """Return the cosine of the expression."""

    @abstractmethod
    def exp(self) -> ExpressionWithTrigonometry:
        """Return the exponential of the expression."""


class Parameter(Expression):
    """Abstract class for substituable parameter."""


class PlaceholderOperationError(ValueError):
    """Error raised when an operation is not supported by the placeholder."""

    def __init__(self) -> None:
        """Instantiate the error."""
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
        """Add `d` to the expression."""
        return AffineExpression(a=self.a, x=self.x, b=self.b + d)

    def _scale_non_null(self, k: float) -> AffineExpression:
        return AffineExpression(a=k * self.a, x=self.x, b=k * self.b)

    def scale(self, k: float) -> ExpressionOrFloat:
        """Multiply the expression by `k`."""
        if k == 0:
            return 0
        return self._scale_non_null(k)

    def __mul__(self, other: object) -> ExpressionOrFloat:
        """Look to the documentation in the parent class."""
        if isinstance(other, SupportsFloat):
            return self.scale(float(other))
        return NotImplemented

    def __rmul__(self, other: object) -> ExpressionOrFloat:
        """Look to the documentation in the parent class."""
        if isinstance(other, SupportsFloat):
            return self.scale(float(other))
        return NotImplemented

    def __add__(self, other: object) -> ExpressionOrFloat:
        """Look to the documentation in the parent class."""
        if isinstance(other, SupportsFloat):
            return self.offset(float(other))
        if isinstance(other, AffineExpression):
            if other.x != self.x:
                raise PlaceholderOperationError
            a = self.a + other.a
            if a == 0:
                return 0
            return AffineExpression(a=a, x=self.x, b=self.b + other.b)
        return NotImplemented

    def __radd__(self, other: object) -> ExpressionOrFloat:
        """Look to the documentation in the parent class."""
        if isinstance(other, SupportsFloat):
            return self.offset(float(other))
        return NotImplemented

    def __sub__(self, other: object) -> ExpressionOrFloat:
        """Look to the documentation in the parent class."""
        if isinstance(other, AffineExpression):
            return self + -other
        if isinstance(other, SupportsFloat):
            return self + -float(other)
        return NotImplemented

    def __rsub__(self, other: object) -> ExpressionOrFloat:
        """Look to the documentation in the parent class."""
        if isinstance(other, SupportsFloat):
            return self._scale_non_null(-1).offset(float(other))
        return NotImplemented

    def __neg__(self) -> ExpressionOrFloat:
        """Look to the documentation in the parent class."""
        return self._scale_non_null(-1)

    def __truediv__(self, other: object) -> ExpressionOrFloat:
        """Look to the documentation in the parent class."""
        if isinstance(other, SupportsFloat):
            return self.scale(1 / float(other))
        return NotImplemented

    def __str__(self) -> str:
        """Return a textual representation of the expression."""
        return f"{self.a} * {self.x} + {self.b}"

    def __eq__(self, other: object) -> bool:
        """Check if two expressions are equal."""
        if isinstance(other, AffineExpression):
            return self.a == other.a and self.x == other.x and self.b == other.b
        return False

    def evaluate(self, value: ExpressionOrSupportsFloat) -> ExpressionOrFloat:
        """Evaluate the expression at `value`."""
        if isinstance(value, SupportsFloat):
            return self.a * float(value) + self.b
        return self.a * value + self.b

    def subs(self, variable: Parameter, value: ExpressionOrSupportsFloat) -> ExpressionOrComplex:
        """Look to the documentation in the parent class."""
        if variable == self.x:
            return self.evaluate(value)
        return self

    def xreplace(self, assignment: Mapping[Parameter, ExpressionOrSupportsFloat]) -> ExpressionOrComplex:
        """Look to the documentation in the parent class."""
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
        """Return the name of the placeholder."""
        return self.__name

    def __repr__(self) -> str:
        """Return a representation of the placeholder."""
        return f"Placeholder({self.__name!r})"

    def __str__(self) -> str:
        """Return the name of the placeholder."""
        return self.__name

    def __eq__(self, other: object) -> bool:
        """Check if two placeholders are identical."""
        if isinstance(other, Parameter):
            return self is other
        return super().__eq__(other)

    def __hash__(self) -> int:
        """Return an hash value for the placeholder."""
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


@overload
def subs(value: Expression, variable: Parameter, substitute: ExpressionOrSupportsFloat) -> Expression: ...


@overload
def subs(value: T, variable: Parameter, substitute: ExpressionOrSupportsFloat) -> T | complex: ...


# The return type could be `T | complex` since `subs` returns `Expression` only
# if `T == Expression`, but `mypy` does not handle this yet: https://github.com/python/mypy/issues/12989
def subs(value: T, variable: Parameter, substitute: ExpressionOrSupportsFloat) -> T | Expression | complex:
    """
    Substitute in `value`.

    If `value` is in instance of :class:`Expression`, then return
    `value.subs(variable, substitute)` (coerced into a complex or a
    float if the result is a number).

    If `value` does not implement `subs`, `value` is returned
    unchanged.

    This function is used to apply substitution to collections where
    some elements are `Expression` and other elements are just
    plain numbers.

    """
    if not isinstance(value, Expression):
        return value
    new_value = value.subs(variable, substitute)
    # On Python<=3.10, complex is not a subtype of SupportsComplex
    if isinstance(new_value, (complex, SupportsComplex)):
        c = complex(new_value)
        if c.imag == 0.0:
            return c.real
        return c
    return new_value


# The return type could be `T | Expression | complex` since `subs` returns `Expression` only
# if `T == Expression`, but `mypy` does not handle this yet: https://github.com/python/mypy/issues/12989
def xreplace(value: T, assignment: Mapping[Parameter, ExpressionOrSupportsFloat]) -> T | Expression | complex:
    """
    Substitute in parallel in `value`.

    If `value` is an an instance of :class:`Expression`, then return
    `value.xreplace(assignment)` (coerced into a complex if the result
    is a number).

    If `value` does not implement `xreplace`, `value` is returned
    unchanged.

    This function is used to apply parallel substitutions to
    collections where some elements are Expression and other elements
    are just plain numbers.

    """
    if not isinstance(value, Expression):
        return value
    new_value = value.xreplace(assignment)
    # On Python<=3.10, complex is not a subtype of SupportsComplex
    if isinstance(new_value, (complex, SupportsComplex)):
        c = complex(new_value)
        if c.imag == 0.0:
            return c.real
        return c
    return new_value


def cos_sin(angle: ExpressionOrFloat) -> tuple[ExpressionOrFloat, ExpressionOrFloat]:
    """Cosine and sine of a float or an expression."""
    if isinstance(angle, Expression):
        if isinstance(angle, ExpressionWithTrigonometry):
            cos: ExpressionOrFloat = angle.cos()
            sin: ExpressionOrFloat = angle.sin()
        else:
            raise PlaceholderOperationError
    else:
        cos = math.cos(angle)
        sin = math.sin(angle)
    return cos, sin


def exp(z: ExpressionOrComplex) -> ExpressionOrComplex:
    """Exponential of a number or an expression."""
    if isinstance(z, Expression):
        if isinstance(z, ExpressionWithTrigonometry):
            return z.exp()
        raise PlaceholderOperationError
    e = np.exp(z)
    # Result type of np.exp is Any!
    assert isinstance(e, (complex, float))
    return e
