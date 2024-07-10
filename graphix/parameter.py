"""Parameter class

Parameter object acts as a placeholder of measurement angles and
allows the manipulation of the measurement pattern without specific
value assignment.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numbers

class Expression(ABC):
    """Expression with parameters."""

    @abstractmethod
    def __mul__(self, other) -> ExpressionOperatorResult: ...

    @abstractmethod
    def __rmul__(self, other) -> ExpressionOperatorResult: ...

    @abstractmethod
    def __add__(self, other) -> ExpressionOperatorResult: ...

    @abstractmethod
    def __radd__(self, other) -> ExpressionOperatorResult: ...

    @abstractmethod
    def __sub__(self, other) -> ExpressionOperatorResult: ...

    @abstractmethod
    def __rsub__(self, other) -> ExpressionOperatorResult: ...

    @abstractmethod
    def __pow__(self, other) -> ExpressionOperatorResult: ...

    @abstractmethod
    def __rpow__(self, other) -> ExpressionOperatorResult: ...

    @abstractmethod
    def __neg__(self) -> ExpressionOperatorResult: ...

    @abstractmethod
    def __truediv__(self, other) -> ExpressionOperatorResult: ...

    @abstractmethod
    def __rtruediv__(self, other) -> ExpressionOperatorResult: ...

    @abstractmethod
    def __mod__(self, other) -> ExpressionOperatorResult: ...

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

    @abstractmethod
    def flatten(self) -> ExpressionOrNumber: ...

    @abstractmethod
    def conj(self) -> ExpressionOrNumber: ...


class PlaceholderOperationError(ValueError):
    def __init__(self):
        super().__init__("Placeholder angles do not support any form of computation before substitution. Either use `subst` with an actual value before the computation, or use a symbolic parameter implementation, such that https://github.com/TeamGraphix/graphix-symbolic .")


class Parameter(Expression):
    """Abstract class for substituable parameter."""
    ...

class Placeholder(Parameter):
    """Placeholder for measurement angles, which allows the pattern optimizations
    without specifying measurement angles for measurement commands.
    Either use for rotation gates of :class:`Circuit` class or for
    the measurement angle of the measurement commands to be added with :meth:`Pattern.add` method.
    """

    def __init__(self, name: str) -> None:
        """Create a new :class:`Placeholder` object.

        Parameters
        ----------
        name : str
            name of the parameter, used for binding values.
        """
        self.__name = name

    @property
    def name(self) -> str:
        return self.__name

    def __repr__(self) -> str:
        return self.__name

    def __str__(self) -> str:
        return self.__name

    def subs(self, variable: Parameter, value: ExpressionOrNumber) -> ExpressionOrNumber:
        if self == variable:
            if isinstance(value, numbers.Number):
                return complex(value)
            return value
        else:
            return self

    def __mul__(self, other) -> ExpressionOrNumber:
        return NotImplemented

    def __rmul__(self, other) -> ExpressionOrNumber:
        return NotImplemented

    def __add__(self, other) -> ExpressionOrNumber:
        return NotImplemented

    def __radd__(self, other) -> ExpressionOrNumber:
        return NotImplemented

    def __sub__(self, other) -> ExpressionOrNumber:
        return NotImplemented

    def __rsub__(self, other) -> ExpressionOrNumber:
        return NotImplemented

    def __pow__(self, other) -> ExpressionOrNumber:
        return NotImplemented

    def __rpow__(self, other) -> ExpressionOrNumber:
        return NotImplemented

    def __neg__(self) -> ExpressionOrNumber:
        return NotImplemented

    def __truediv__(self, other) -> ExpressionOrNumber:
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

    def subs(self, variable: Parameter, value: ExpressionOrNumber) -> ExpressionOrNumber:
        if variable is self:
            return value
        return self

    def flatten(self) -> ExpressionOrNumber:
        raise PlaceholderOperationError()

    def conj(self) -> ExpressionOrNumber:
        raise PlaceholderOperationError()


ExpressionOrNumber = Expression | numbers.Number

ExpressionOperatorResult = ExpressionOrNumber | type(NotImplemented)


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
        if isinstance(new_value, numbers.Number):
            return complex(new_value)
        return new_value
    return value
