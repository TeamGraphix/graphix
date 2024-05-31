"""Parameter class

Parameter object acts as a placeholder of measurement angles and
allows the manipulation of the measurement pattern without specific
value assignment.

"""

from __future__ import annotations

import numbers

import numpy as np
import sympy as sp


class ParameterExpression:
    """Expression with parameters.

    Implements arithmetic operations. This is essentially a wrapper over
    sp.Expr, exposing methods like cos, conjugate, etc., that are
    expected by the simulator back-ends.
    """

    def __init__(self, expression: sp.Expr):
        assert isinstance(expression, sp.Expr)
        self._expression = expression

    def __mul__(self, other) -> ParameterExpression:
        if isinstance(other, numbers.Number):
            return ParameterExpression(self._expression * other)
        elif isinstance(other, ParameterExpression):
            return ParameterExpression(self._expression * other._expression)
        else:
            return NotImplemented

    def __rmul__(self, other) -> ParameterExpression:
        if isinstance(other, numbers.Number):
            return ParameterExpression(other * self._expression)
        elif isinstance(other, ParameterExpression):
            return ParameterExpression(other._expression * self._expression)
        else:
            return NotImplemented

    def __add__(self, other) -> ParameterExpression:
        if isinstance(other, numbers.Number):
            return ParameterExpression(self._expression + other)
        elif isinstance(other, ParameterExpression):
            return ParameterExpression(self._expression + other._expression)
        else:
            return NotImplemented

    def __radd__(self, other) -> ParameterExpression:
        if isinstance(other, numbers.Number):
            return ParameterExpression(other + self._expression)
        elif isinstance(other, ParameterExpression):
            return ParameterExpression(other._expression + self._expression)
        else:
            return NotImplemented

    def __sub__(self, other) -> ParameterExpression:
        if isinstance(other, numbers.Number):
            return ParameterExpression(self._expression - other)
        elif isinstance(other, ParameterExpression):
            return ParameterExpression(self._expression - other._expression)
        else:
            return NotImplemented

    def __rsub__(self, other) -> ParameterExpression:
        if isinstance(other, numbers.Number):
            return ParameterExpression(other - self._expression)
        elif isinstance(other, ParameterExpression):
            return ParameterExpression(other._expression - self._expression)
        else:
            return NotImplemented

    def __neg__(self) -> ParameterExpression:
        return ParameterExpression(-self._expression)

    def __truediv__(self, other) -> ParameterExpression:
        if isinstance(other, numbers.Number):
            return ParameterExpression(self._expression / other)
        elif isinstance(other, ParameterExpression):
            return ParameterExpression(self._expression / other._expression)
        else:
            return NotImplemented

    def __rtruediv__(self, other) -> ParameterExpression:
        if isinstance(other, numbers.Number):
            return ParameterExpression(other / self._expression)
        elif isinstance(other, ParameterExpression):
            return ParameterExpression(other._expression / self._expression)
        else:
            return NotImplemented

    def __mod__(self, other) -> float:
        """mod magic function returns nan so that evaluation of
        mod of measurement angles in :meth:`graphix.pattern.is_pauli_measurement`
        will not cause error. returns nan so that this will not be considered Pauli measurement.
        """
        assert isinstance(other, float) or isinstance(other, int)
        return np.nan

    def sin(self) -> ParameterExpression:
        return ParameterExpression(sp.sin(self._expression))

    def cos(self) -> ParameterExpression:
        return ParameterExpression(sp.cos(self._expression))

    def tan(self) -> ParameterExpression:
        return ParameterExpression(sp.tan(self._expression))

    def arcsin(self) -> ParameterExpression:
        return ParameterExpression(sp.asin(self._expression))

    def arccos(self) -> ParameterExpression:
        return ParameterExpression(sp.acos(self._expression))

    def arctan(self) -> ParameterExpression:
        return ParameterExpression(sp.atan(self._expression))

    def exp(self) -> ParameterExpression:
        return ParameterExpression(sp.exp(self._expression))

    def log(self) -> ParameterExpression:
        return ParameterExpression(sp.log(self._expression))

    def conjugate(self) -> ParameterExpression:
        return ParameterExpression(sp.conjugate(self._expression))

    def sqrt(self) -> ParameterExpression:
        return ParameterExpression(sp.sqrt(self._expression))

    @property
    def expression(self) -> sp.Expr:
        return self._expression

    def __repr__(self) -> str:
        return str(self._expression)

    def subs(self, variable, value) -> ParameterExpression | numbers.Number:
        result = sp.simplify(self._expression.subs(variable._expression, value))
        if isinstance(result, numbers.Number):
            return result
        else:
            return ParameterExpression(result)


class Parameter(ParameterExpression):
    """Placeholder for measurement angles, which allows the pattern optimizations
    without specifying measurement angles for measurement commands.
    Either use for rotation gates of :class:`Circuit` class or for
    the measurement angle of the measurement commands to be added with :meth:`Pattern.add` method.
    Example:
    .. code-block:: python

        # rotation gate
        from graphix import Circuit
        circuit = Circuit(1)
        alpha = Parameter('alpha')
        circuit.rx(0, alpha)
        pattern = circuit.transpile()
        # simulate with parameter assignment
        sv = pattern.subs(alpha, 0.5).simulate_pattern()
        # simulate without pattern assignment
        # (the resulting state vector is symbolic)
        # Note: pr_calc=False is mandatory since we cannot compute probabilities on
        # symbolic states; we explore one arbitrary branch, and the result is only
        # well-defined if the pattern is deterministic.
        sv2 = pattern.simulate_pattern(pr_calc=False)
        # Substituting alpha in the resulting state vector should yield the same result
        assert np.allclose(sv.psi, sv2.subs(alpha, 0.5).psi)
    """

    def __init__(self, name):
        """Create a new :class:`Parameter` object.

        Parameters
        ----------
        name : str
            name of the parameter, used for binding values.
        """
        assert isinstance(name, str)
        self._name = name
        super().__init__(sp.Symbol(name=name))

    @property
    def name(self):
        return self._name


def subs(value, variable, substitute):
    """Generic substitution in `value`: if `value` implements the
    method `subs`, then return `value.subs(variable, substitute)`
    (coerced into a complex if the result is a number).  If `value`
    does not implement `subs`, `value` is returned unchanged.

    This function is used to apply substitution to collections where
    some elements are ParameterExpression and other elements are just
    plain numbers.

    """
    subs = getattr(value, "subs", None)
    if subs:
        new_value = subs(variable, substitute)
        if isinstance(new_value, numbers.Number):
            return complex(new_value)
        return new_value
    return value
