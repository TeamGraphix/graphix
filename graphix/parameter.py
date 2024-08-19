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
from numbers import Number
from typing import TYPE_CHECKING

import pydantic_core

if TYPE_CHECKING:
    import pydantic


class Expression(ABC):
    """Expression with parameters."""

    @abstractmethod
    def __mul__(self, other) -> ExpressionOrNumber: ...

    @abstractmethod
    def __rmul__(self, other) -> ExpressionOrNumber: ...

    @abstractmethod
    def __add__(self, other) -> ExpressionOrNumber: ...

    @abstractmethod
    def __radd__(self, other) -> ExpressionOrNumber: ...

    @abstractmethod
    def __sub__(self, other) -> ExpressionOrNumber: ...

    @abstractmethod
    def __rsub__(self, other) -> ExpressionOrNumber: ...

    @abstractmethod
    def __pow__(self, other) -> ExpressionOrNumber: ...

    @abstractmethod
    def __rpow__(self, other) -> ExpressionOrNumber: ...

    @abstractmethod
    def __neg__(self) -> ExpressionOrNumber: ...

    @abstractmethod
    def __truediv__(self, other) -> ExpressionOrNumber: ...

    @abstractmethod
    def __rtruediv__(self, other) -> ExpressionOrNumber: ...

    @abstractmethod
    def __mod__(self, other) -> ExpressionOrNumber: ...

    @abstractmethod
    def sin(self) -> ExpressionOrNumber: ...

    @abstractmethod
    def cos(self) -> ExpressionOrNumber: ...

    @abstractmethod
    def tan(self) -> ExpressionOrNumber: ...

    @abstractmethod
    def arcsin(self) -> ExpressionOrNumber: ...

    @abstractmethod
    def arccos(self) -> ExpressionOrNumber: ...

    @abstractmethod
    def arctan(self) -> ExpressionOrNumber: ...

    @abstractmethod
    def exp(self) -> ExpressionOrNumber: ...

    @abstractmethod
    def log(self) -> ExpressionOrNumber: ...

    @abstractmethod
    def conjugate(self) -> ExpressionOrNumber: ...

    @abstractmethod
    def sqrt(self) -> ExpressionOrNumber: ...

    @abstractmethod
    def __repr__(self) -> str: ...

    @abstractmethod
    def __str__(self) -> str: ...

    @abstractmethod
    def subs(self, variable: Parameter, value: ExpressionOrNumber) -> ExpressionOrNumber: ...

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: typing.Any, handler: pydantic.GetCoreSchemaHandler
    ) -> pydantic_core.CoreSchema:
        def check_expression(obj) -> Expression:
            if not isinstance(obj, Expression):
                raise ValueError("Expression expected")
            return obj

        return pydantic_core.core_schema.no_info_plain_validator_function(function=check_expression)


class Parameter(Expression):
    """Abstract class for substituable parameter."""

    ...


class PlaceholderOperationError(ValueError):
    def __init__(self):
        super().__init__(
            "Placeholder angles do not support any form of computation before substitution except affine operation. Either use `subs` with an actual value before the computation, or use a symbolic parameter implementation, such as https://github.com/TeamGraphix/graphix-symbolic ."
        )


@dataclass
class AffineExpression(Expression):
    """Affine expression.

    An affine expression is of the form `a*x+b` where `a` and `b` are numbers and `x` is a parameter.
    """

    a: Number
    x: Parameter
    b: Number

    def offset(self, d: Number) -> AffineExpression:
        return AffineExpression(a=self.a, x=self.x, b=self.b + d)

    def scale(self, k: Number) -> ExpressionOrNumber:
        if k == 0:
            return 0
        return AffineExpression(a=k * self.a, x=self.x, b=k * self.b)

    def __mul__(self, other) -> ExpressionOrNumber:
        if isinstance(other, Number):
            return self.scale(other)
        return NotImplemented

    def __rmul__(self, other) -> ExpressionOrNumber:
        if isinstance(other, Number):
            return self.scale(other)
        return NotImplemented

    def __add__(self, other) -> ExpressionOrNumber:
        if isinstance(other, Number):
            return self.offset(other)
        if isinstance(other, AffineExpression):
            if other.x != self.x:
                raise PlaceholderOperationError()
            a = self.a + other.a
            if a == 0:
                return 0
            return AffineExpression(a=a, x=self.x, b=self.b + other.b)
        return NotImplemented

    def __radd__(self, other) -> ExpressionOrNumber:
        if isinstance(other, Number):
            return self.offset(other)
        return NotImplemented

    def __sub__(self, other) -> ExpressionOrNumber:
        if isinstance(other, (AffineExpression, Number)):
            return self + -other
        return NotImplemented

    def __rsub__(self, other) -> ExpressionOrNumber:
        if isinstance(other, Number):
            return self.scale(-1).offset(other)
        return NotImplemented

    def __pow__(self, other) -> ExpressionOrNumber:
        return NotImplemented

    def __rpow__(self, other) -> ExpressionOrNumber:
        return NotImplemented

    def __neg__(self) -> ExpressionOrNumber:
        return self.scale(-1)

    def __truediv__(self, other) -> ExpressionOrNumber:
        if isinstance(other, Number):
            return self.scale(1 / other)
        return NotImplemented

    def __rtruediv__(self, other) -> ExpressionOrNumber:
        return NotImplemented

    def __mod__(self, other) -> ExpressionOrNumber:
        return NotImplemented

    def sin(self) -> ExpressionOrNumber:
        raise PlaceholderOperationError()

    def cos(self) -> ExpressionOrNumber:
        raise PlaceholderOperationError()

    def tan(self) -> ExpressionOrNumber:
        raise PlaceholderOperationError()

    def arcsin(self) -> ExpressionOrNumber:
        raise PlaceholderOperationError()

    def arccos(self) -> ExpressionOrNumber:
        raise PlaceholderOperationError()

    def arctan(self) -> ExpressionOrNumber:
        raise PlaceholderOperationError()

    def exp(self) -> ExpressionOrNumber:
        raise PlaceholderOperationError()

    def log(self) -> ExpressionOrNumber:
        raise PlaceholderOperationError()

    def conjugate(self) -> ExpressionOrNumber:
        raise PlaceholderOperationError()

    def sqrt(self) -> ExpressionOrNumber:
        raise PlaceholderOperationError()

    def __str__(self) -> str:
        return f"{self.a} * {self.x} + {self.b}"

    def __eq__(self, other) -> bool:
        if isinstance(other, AffineExpression):
            return self.a == other.a and self.x == other.x and self.b == other.b
        return False

    def subs(self, variable: Parameter, value: ExpressionOrNumber) -> ExpressionOrNumber:
        if variable == self.x:
            return self.a * value + self.b
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
    computation and are not suitable for simulation.  Either use
    :func:`Circuit.subs` or :func:`Pattern.subs` with an actual value
    before the computation, or use a symbolic parameter
    implementation, such as
    https://github.com/TeamGraphix/graphix-symbolic .

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

    def __str__(self) -> str:
        return self.__name

    def __eq__(self, other) -> bool:
        if isinstance(other, Parameter):
            return self is other
        return super().__eq__(other)


if sys.version_info >= (3, 10):
    ExpressionOrFloat = Expression | float

    ExpressionOrNumber = Expression | Number
else:
    ExpressionOrFloat = typing.Union[Expression, float]

    ExpressionOrNumber = typing.Union[Expression, Number]


def subs(value, variable: Parameter, substitute: ExpressionOrNumber):
    """Generic substitution in `value`: if `value` implements the
    method `subs`, then return `value.subs(variable, substitute)`
    (coerced into a complex if the result is a number).  If `value`
    does not implement `subs`, `value` is returned unchanged.

    This function is used to apply substitution to collections where
    some elements are Expression and other elements are just
    plain numbers.

    """
    subs = getattr(value, "subs", None)
    if subs:
        new_value = subs(variable, substitute)
        if isinstance(new_value, Number):
            return complex(new_value)
        return new_value
    return value
