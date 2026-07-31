"""Parameter class.

Parameter object acts as a placeholder of measurement angles and
allows the manipulation of the measurement pattern without specific
value assignment.

"""

from __future__ import annotations

import cmath
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, SupportsComplex, SupportsFloat, TypeAlias, TypeVar, overload

# override introduced in Python 3.12
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Self


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
    def with_parameter(self, variable: Parameter, substitute: ExpressionOrSupportsFloat) -> ExpressionOrComplex:
        """Substitute a parameter with a value or expression."""

    @abstractmethod
    def with_parameters(self, assignment: Mapping[Parameter, ExpressionOrSupportsFloat]) -> ExpressionOrComplex:
        """Perform parallel substitution of multiple parameters."""

    def subs(self, variable: Parameter, substitute: ExpressionOrSupportsFloat) -> ExpressionOrComplex:
        """Alias for :meth:`with_parameter`, following the ``sympy`` convention."""
        return self.with_parameter(variable, substitute)

    def xreplace(self, assignment: Mapping[Parameter, ExpressionOrSupportsFloat]) -> ExpressionOrComplex:
        """Alias for :meth:`with_parameters`, following the ``sympy`` convention."""
        return self.with_parameters(assignment)


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

    An affine expression is of the form *a*x+b* where *a* and *b* are numbers and *x* is a parameter.
    """

    a: float
    x: Parameter
    b: float

    def offset(self, d: float) -> AffineExpression:
        """Add *d* to the expression."""
        return AffineExpression(a=self.a, x=self.x, b=self.b + d)

    def scale_non_null(self, k: float) -> AffineExpression:
        """Return ``self`` scaled by ``k`` assuming ``k`` is non-zero.

        Parameters
        ----------
        k : float
            Scaling factor.

        Returns
        -------
        AffineExpression
            A new expression scaled by ``k``.
        """
        return AffineExpression(a=k * self.a, x=self.x, b=k * self.b)

    def scale(self, k: float) -> ExpressionOrFloat:
        """Multiply the expression by `k`."""
        if k == 0:
            return 0
        return self.scale_non_null(k)

    @override
    def __mul__(self, other: object) -> ExpressionOrFloat:
        """Look to the documentation in the parent class."""
        if isinstance(other, SupportsFloat):
            return self.scale(float(other))
        return NotImplemented

    @override
    def __rmul__(self, other: object) -> ExpressionOrFloat:
        """Look to the documentation in the parent class."""
        if isinstance(other, SupportsFloat):
            return self.scale(float(other))
        return NotImplemented

    @override
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

    @override
    def __radd__(self, other: object) -> ExpressionOrFloat:
        """Look to the documentation in the parent class."""
        if isinstance(other, SupportsFloat):
            return self.offset(float(other))
        return NotImplemented

    @override
    def __sub__(self, other: object) -> ExpressionOrFloat:
        """Look to the documentation in the parent class."""
        if isinstance(other, AffineExpression):
            return self + -other
        if isinstance(other, SupportsFloat):
            return self + -float(other)
        return NotImplemented

    @override
    def __rsub__(self, other: object) -> ExpressionOrFloat:
        """Look to the documentation in the parent class."""
        if isinstance(other, SupportsFloat):
            return self.scale_non_null(-1).offset(float(other))
        return NotImplemented

    @override
    def __neg__(self) -> ExpressionOrFloat:
        """Look to the documentation in the parent class."""
        return self.scale_non_null(-1)

    @override
    def __truediv__(self, other: object) -> ExpressionOrFloat:
        """Look to the documentation in the parent class."""
        if isinstance(other, SupportsFloat):
            return self.scale(1 / float(other))
        return NotImplemented

    @override
    def __str__(self) -> str:
        """Return a textual representation of the expression."""
        return f"{self.a} * {self.x} + {self.b}"

    @override
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

    @override
    def with_parameter(
        self, variable: Parameter, substitute: ExpressionOrSupportsFloat
    ) -> AffineExpression | ExpressionOrComplex:
        """Look to the documentation in the parent class."""
        if variable == self.x:
            return self.evaluate(substitute)
        return self

    @override
    def with_parameters(self, assignment: Mapping[Parameter, ExpressionOrSupportsFloat]) -> ExpressionOrComplex:
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

    @override
    def __repr__(self) -> str:
        """Return a representation of the placeholder."""
        return f"Placeholder({self.__name!r})"

    @override
    def __str__(self) -> str:
        """Return the name of the placeholder."""
        return self.__name

    @override
    def __eq__(self, other: object) -> bool:
        """Check if two placeholders are identical."""
        if isinstance(other, Parameter):
            return self is other
        return super().__eq__(other)

    @override
    def __hash__(self) -> int:
        """Return an hash value for the placeholder."""
        return id(self)


ExpressionOrFloat: TypeAlias = Expression | float

ExpressionOrComplex: TypeAlias = Expression | complex

ExpressionOrSupportsFloat: TypeAlias = Expression | SupportsFloat

# `ExpressionOrSupportsComplex` is based on
# `ExpressionOrSupportsFloat` rather than `Expression`, because `int`
# and `float` implement `SupportsFloat` but not `SupportsComplex`,
# even though `complex(int)` and `complex(float)` are valid.
ExpressionOrSupportsComplex: TypeAlias = ExpressionOrSupportsFloat | SupportsComplex


_T = TypeVar("_T")


def check_expression_or_complex(value: object) -> ExpressionOrComplex:
    """Check that the given object is of type ExpressionOrComplex and return it."""
    if isinstance(value, Expression):
        return value
    if isinstance(value, SupportsComplex):
        return complex(value)
    msg = f"ExpressionOrComplex expected, but {type(value)} found."
    raise TypeError(msg)


def check_expression_or_float(value: object) -> ExpressionOrFloat:
    """Check that the given object is of type ExpressionOrFloat and return it."""
    if isinstance(value, Expression):
        return value
    if isinstance(value, SupportsFloat):
        return float(value)
    msg = f"ExpressionOrFloat expected, but {type(value)} found."
    raise TypeError(msg)


@overload
def with_parameter(
    value: ExpressionOrFloat, variable: Parameter, substitute: ExpressionOrSupportsFloat
) -> ExpressionOrFloat: ...


@overload
def with_parameter(value: _T, variable: Parameter, substitute: ExpressionOrSupportsFloat) -> _T | complex: ...


# The return type could be `_T | complex` since `with_parameter` returns `Expression` only
# if `_T == Expression`, but `mypy` does not handle this yet: https://github.com/python/mypy/issues/12989
def with_parameter(value: _T, variable: Parameter, substitute: ExpressionOrSupportsFloat) -> _T | Expression | complex:
    """
    Substitute in `value`.

    If `value` is in instance of :class:`Expression`, then return
    `value.with_parameter(variable, substitute)` (coerced into a complex or a
    float if the result is a number).

    If ``value`` does not implement ``with_parameter``, ``value`` is returned
    unchanged.

    This function is used to apply substitution to collections where
    some elements are `Expression` and other elements are just
    plain numbers.

    """
    if not isinstance(value, Expression):
        return value
    new_value = value.with_parameter(variable, substitute)
    if isinstance(new_value, complex) and math.isclose(new_value.imag, 0.0):
        # Conversion to float, to enable the simulator to call
        # real trigonometric functions to the result.
        return new_value.real
    return new_value


@overload
def with_parameters(
    value: ExpressionOrFloat, assignment: Mapping[Parameter, ExpressionOrSupportsFloat]
) -> ExpressionOrFloat: ...


@overload
def with_parameters(
    value: _T, assignment: Mapping[Parameter, ExpressionOrSupportsFloat]
) -> _T | Expression | complex: ...


# The return type could be `_T | Expression | complex` since `subs` returns `Expression` only
# if `_T == Expression`, but `mypy` does not handle this yet: https://github.com/python/mypy/issues/12989
def with_parameters(value: _T, assignment: Mapping[Parameter, ExpressionOrSupportsFloat]) -> _T | Expression | complex:
    """
    Substitute in parallel in `value`.

    If `value` is an an instance of :class:`Expression`, then return
    `value.with_parameters(assignment)` (coerced into a complex if the result
    is a number).

    If `value` does not implement `xreplace`, `value` is returned
    unchanged.

    This function is used to apply parallel substitutions to
    collections where some elements are Expression and other elements
    are just plain numbers.

    """
    if not isinstance(value, Expression):
        return value
    new_value = value.with_parameters(assignment)
    if isinstance(new_value, complex) and math.isclose(new_value.imag, 0.0):
        # Conversion to float, to enable the simulator to call
        # real trigonometric functions to the result.
        return new_value.real
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
    return cmath.exp(z)


class Parameterizable(ABC):
    """Abstract base class for objects that hold parameters.

    This class defines the abstract methods :meth:`with_parameter` for
    single-parameter substitution and :meth:`with_parameters` for parallel
    substitution of multiple parameters.

    It also exposes :meth:`subs` and :meth:`xreplace` as aliases for
    :meth:`with_parameter` and :meth:`with_parameters`, respectively,
    following the ``sympy`` convention.
    """

    @abstractmethod
    def with_parameter(self, variable: Parameter, substitute: ExpressionOrSupportsFloat) -> Self:
        """Substitute a parameter with a value or expression."""

    @abstractmethod
    def with_parameters(self, assignment: Mapping[Parameter, ExpressionOrSupportsFloat]) -> Self:
        """Perform parallel substitution of multiple parameters."""

    def subs(self, variable: Parameter, substitute: ExpressionOrSupportsFloat) -> Self:
        """Alias for :meth:`with_parameter`, following the ``sympy`` convention."""
        return self.with_parameter(variable, substitute)

    def xreplace(self, assignment: Mapping[Parameter, ExpressionOrSupportsFloat]) -> Self:
        """Alias for :meth:`with_parameters`, following the ``sympy`` convention."""
        return self.with_parameters(assignment)


class InplaceParameterizable(Parameterizable, ABC):
    """Abstract base class for objects that hold parameters and can be modified in place.

    This class defines the abstract methods :meth:`replace_parameter`
    for single-parameter substitution and :meth:`replace_parameters`
    for parallel substitution of multiple parameters. By default, these
    methods modify the object in place, with an optional ``copy``
    parameter. If ``copy=True``, a modified copy is returned instead.

    The methods :meth:`with_parameter` and :meth:`with_parameters` are
    equivalent to calling :meth:`replace_parameter` and
    :meth:`replace_parameters`, respectively, with ``copy=True``.
    """

    @abstractmethod
    def replace_parameter(
        self, variable: Parameter, substitute: ExpressionOrSupportsFloat, *, copy: bool = False
    ) -> Self:
        """Substitute all occurrences of the given variable in measurement angles by the given value.

        Parameters
        ----------
        variable : Parameter
            The symbolic expression to be replaced within the measurement angles.
        substitute : ExpressionOrSupportsFloat
            The value or symbolic expression to substitute in place of ``variable``.

        copy : bool, optional
            If ``True``, the current object remains unchanged and a
            new object is returned. The default is ``False``, meaning
            that changes are performed in place.

        Returns
        -------
        Self
            The object with the updated measurement parameters.
            Equal to ``self`` if ``copy`` is ``False``.
        """

    @abstractmethod
    def replace_parameters(
        self, assignment: Mapping[Parameter, ExpressionOrSupportsFloat], *, copy: bool = False
    ) -> Self:
        """Substitute all occurrences of the given keys in measurement angles by the given values in parallel.

        Parameters
        ----------
        assignment : Mapping[Parameter, ExpressionOrSupportsFloat]
            A dictionary-like mapping where keys are the `Parameter` objects to be replaced and values are the new expressions or numerical values.

        copy : bool, optional
            If ``True``, the current object remains unchanged and a
            new object is returned. The default is ``False``, meaning
            that changes are performed in place.

        Returns
        -------
        Self
            The object with the updated measurement parameters.
            Equal to ``self`` if ``copy`` is ``False``.
        """

    @override
    def with_parameter(self, variable: Parameter, substitute: ExpressionOrSupportsFloat) -> Self:
        return self.replace_parameter(variable, substitute, copy=True)

    @override
    def with_parameters(self, assignment: Mapping[Parameter, ExpressionOrSupportsFloat]) -> Self:
        return self.replace_parameters(assignment, copy=True)
