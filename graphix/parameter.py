"""Parameter class

Parameter object acts as a placeholder of measurement angles and
allows the manipulation of the measurement pattern without specific
value assignment.

"""

from __future__ import annotations

import cmath
import functools
import numbers
from enum import Enum

import numpy as np


class Expression:
    """Expression with parameters."""

    def __mul__(self, other) -> Expression | complex | type(NotImplemented):
        if isinstance(other, numbers.Number):
            return Mul(self, complex(other)).simplify()
        elif isinstance(other, Expression):
            return Mul(self, other).simplify()
        else:
            return NotImplemented

    def __rmul__(self, other) -> Expression | complex | type(NotImplemented):
        if isinstance(other, numbers.Number):
            return Mul(complex(other), self).simplify()
        elif isinstance(other, Expression):
            return Mul(other, self).simplify()
        else:
            return NotImplemented

    def __add__(self, other) -> Expression | complex | type(NotImplemented):
        if isinstance(other, numbers.Number):
            return Add(self, complex(other)).simplify()
        elif isinstance(other, Expression):
            return Add(self, other).simplify()
        else:
            return NotImplemented

    def __radd__(self, other) -> Expression | complex | type(NotImplemented):
        if isinstance(other, numbers.Number):
            return Add(complex(other), self).simplify()
        elif isinstance(other, Expression):
            return Add(other, self).simplify()
        else:
            return NotImplemented

    def __sub__(self, other) -> Expression | complex | type(NotImplemented):
        if isinstance(other, numbers.Number):
            return Sub(self, complex(other)).simplify()
        elif isinstance(other, Expression):
            return Sub(self, other).simplify()
        else:
            return NotImplemented

    def __rsub__(self, other) -> Expression | complex | type(NotImplemented):
        if isinstance(other, numbers.Number):
            return Sub(complex(other), self).simplify()
        elif isinstance(other, Expression):
            return Sub(other, self).simplify()
        else:
            return NotImplemented

    def __rsub__(self, other) -> Expression | complex | type(NotImplemented):
        if isinstance(other, numbers.Number):
            return Sub(complex(other), self).simplify()
        elif isinstance(other, Expression):
            return Sub(other, self).simplify()
        else:
            return NotImplemented

    def __pow__(self, other) -> Expression | complex | type(NotImplemented):
        if isinstance(other, numbers.Number):
            return Pow(self, complex(other)).simplify()
        elif isinstance(other, Expression):
            return Pow(self, other).simplify()
        else:
            return NotImplemented

    def __rpow__(self, other) -> Expression | complex | type(NotImplemented):
        if isinstance(other, numbers.Number):
            return Pow(complex(other), self).simplify()
        elif isinstance(other, Expression):
            return Pow(other, self).simplify()
        else:
            return NotImplemented

    def __neg__(self) -> Expression | complex:
        return Minus(self).simplify()

    def __truediv__(self, other) -> Expression | complex | type(NotImplemented):
        if isinstance(other, numbers.Number):
            return Div(self, complex(other)).simplify()
        elif isinstance(other, Expression):
            return Div(self, other).simplify()
        else:
            return NotImplemented

    def __rtruediv__(self, other) -> Expression | complex | type(NotImplemented):
        if isinstance(other, numbers.Number):
            return Div(complex(other), self).simplify()
        elif isinstance(other, Expression):
            return Div(other, self).simplify()
        else:
            return NotImplemented

    def __mod__(self, other) -> float:
        """mod magic function returns nan so that evaluation of
        mod of measurement angles in :meth:`graphix.pattern.is_pauli_measurement`
        will not cause error. returns nan so that this will not be considered Pauli measurement.
        """
        return np.nan

    def sin(self) -> Expression | complex:
        return Sin(self).simplify()

    def cos(self) -> Expression | complex:
        return Cos(self).simplify()

    def tan(self) -> Expression | complex:
        return Tan(self).simplify()

    def arcsin(self) -> Expression | complex:
        return ArcSin(self).simplify()

    def arccos(self) -> Expression | complex:
        return ArcCos(self).simplify()

    def arctan(self) -> Expression | complex:
        return ArcTan(self).simplify()

    def exp(self) -> Expression | complex:
        return Exp(self).simplify()

    def log(self) -> Expression | complex:
        return Log(self).simplify()

    def conjugate(self) -> Expression | complex:
        if self.is_float():
            return self
        return Conjugate(self)

    def sqrt(self) -> Expression | complex:
        return Sqrt(self).simplify()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        target = []
        self.format(0, target)
        return "".join(target)

    def format(self, precedence: int, target: list[str]) -> None: ...

    def subs(self, variable: Parameter, value: Expression | numbers.Number) -> Expression | complex: ...

    def simplify(self) -> Expression | complex: ...

    def is_float(self) -> bool:
        return False

    def is_integer(self) -> bool:
        return False

    def flatten(self) -> Expression:
        return self

    def conj(self) -> Expression | complex:
        return self.conjugate()


class Parameter(Expression):
    """Placeholder for measurement angles, which allows the pattern optimizations
    without specifying measurement angles for measurement commands.
    Either use for rotation gates of :class:`Circuit` class or for
    the measurement angle of the measurement commands to be added with :meth:`Pattern.add` method.
    Example:
    .. code-block:: python

        import numpy as np
        from graphix import Circuit
        circuit = Circuit(1)
        alpha = Parameter('alpha')
        # rotation gate
        circuit.rx(0, alpha)
        pattern = circuit.transpile()
        # Both simulations (numeric and symbolic) will use the same
        # seed for random number generation, to ensure that the
        # explored branch is the same for the two simulations.
        seed = np.random.integers(2**63)
        # simulate with parameter assignment
        sv = pattern.subs(alpha, 0.5).simulate_pattern(pr_calc=False, rng=np.random.default_rng(seed))
        # simulate without pattern assignment
        # (the resulting state vector is symbolic)
        # Note: pr_calc=False is mandatory since we cannot compute probabilities on
        # symbolic states; we explore one arbitrary branch.
        sv2 = pattern.simulate_pattern(pr_calc=False, rng=np.random.default_rng(seed))
        # Substituting alpha in the resulting state vector should yield the same result
        assert np.allclose(sv.psi, sv2.subs(alpha, 0.5).psi)
    """

    def __init__(self, name: str, type: type = float) -> None:
        """Create a new :class:`Parameter` object.

        Parameters
        ----------
        name : str
            name of the parameter, used for binding values.
        """
        self.__name = name
        self.__type = type

    @property
    def name(self) -> str:
        return self.__name

    @property
    def type(self) -> type:
        return self.__type

    def subs(self, variable: Parameter, value: Expression | numbers.Number) -> Expression | complex:
        if self == variable:
            if isinstance(value, numbers.Number):
                return complex(value)
            return value
        else:
            return self

    def format(self, precedence: int, target: list[str]):
        target.append(self.name)

    def simplify(self) -> Expression | complex:
        return self

    def is_float(self) -> bool:
        return self.type == int or self.type == float

    def is_integer(self) -> bool:
        return self.type == int


def simplify(expr: Expression | complex) -> Expression | complex:
    if isinstance(expr, complex):
        return expr
    return expr.simplify()


def format_expr(expr: Expression | complex, precedence: int, target: list[str]) -> None:
    if isinstance(expr, complex):
        if expr.imag == 0:
            if expr.real.is_integer():
                target.append(str(int(expr.real)))
            else:
                target.append(str(expr.real))
        else:
            target.append(str(expr))
    else:
        expr.format(precedence, target)


def is_constant_float(expr: Expression | complex) -> bool:
    return isinstance(expr, complex) and expr.imag == 0


def is_float(expr: Expression | complex) -> bool:
    if isinstance(expr, complex):
        return expr.imag == 0
    return expr.is_float()


def is_constant_integer(expr: Expression | complex) -> bool:
    return is_constant_float(expr) and float(expr).is_integer()


def is_integer(expr: Expression | complex) -> bool:
    if isinstance(expr, complex):
        return expr.imag == 0 and float(expr).is_integer()
    return expr.is_integer()


class Compound(Expression):
    def __init__(self, children: typing.Iterable[Expression | complex]) -> None:
        self.__children = tuple(children)
        assert all(isinstance(child, (Expression, complex)) for child in self.__children)

    @classmethod
    def from_iterable(cls, operands: typing.Iterable[Expression | complex]) -> Compound: ...

    @property
    def children(self) -> tuple[Expression | complex, ...]:
        return self.__children

    def __getitem__(self, index) -> Expression | complex:
        return self.__children[index]

    def subs(self, variable: Parameter, value: Expression | numbers.Number) -> Expression | complex:
        return self.__class__.from_iterable((subs(child, variable, value) for child in self.children)).simplify()

    def simplify(self) -> Expression | complex:
        children = [simplify(child) for child in self.children]
        if all(isinstance(child, complex) for child in children):
            return self.eval(children)
        return self.__class__.from_iterable(children)

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Compound)
            and self.__class__ == other.__class__
            and all(child == other_child for child, other_child in zip(self.__children, other.children))
        )

    def __hash__(self) -> int:
        return hash(self.__children)


class Unary(Compound):
    def __init__(self, operand):
        super().__init__((operand,))

    @classmethod
    def from_iterable(cls, operands: typing.Iterable[Expression | complex]):
        operand = next(iter(operands))
        return cls(operand)

    @property
    def operand(self) -> Expression:
        return self[0]


class Minus(Unary):
    @property
    def symbol(self) -> str:
        return "-"

    @property
    def precedence(self) -> int:
        return 3

    def format(self, precedence: int, target: list[str]) -> None:
        need_parentheses = precedence > self.precedence
        if need_parentheses:
            target.append("(")
        target.append("- ")
        format_expr(self.operand, self.precedence, target)
        if need_parentheses:
            target.append(")")

    def eval(self, children: list[complex]) -> complex:
        return -children[0]

    def simplify(self) -> Expression | complex:
        simpl = super().simplify()
        return simplify_linear_combination(simpl)

    def is_float(self) -> bool:
        return self.operand.is_float()

    def is_integer(self) -> bool:
        return self.operand.is_integer()


class Function(Unary):
    def format(self, precedence: int, target: list[str]) -> None:
        target.append(self.symbol)
        target.append("(")
        format_expr(self.operand, 0, target)
        target.append(")")


class Sin(Function):
    @property
    def symbol(self) -> str:
        return "sin"

    def eval(self, children: list[complex]) -> complex:
        argument = children[0]
        return cmath.sin(argument.real)

    def simplify(self) -> Expression | complex:
        simpl = super().simplify()
        if isinstance(simpl, Sin) and isinstance(simpl.operand, Minus):
            return Minus(Sin(simpl.operand.operand))
        return simpl

    def is_float(self) -> bool:
        return self.operand.is_float()


class Cos(Function):
    @property
    def symbol(self) -> str:
        return "cos"

    def eval(self, children: list[complex]) -> complex:
        argument = children[0]
        return cmath.cos(argument.real)

    def simplify(self) -> Expression | complex:
        simpl = super().simplify()
        if isinstance(simpl, Cos) and isinstance(simpl.operand, Minus):
            return Cos(simpl.operand.operand)
        return simpl

    def is_float(self) -> bool:
        return self.operand.is_float()


class Tan(Function):
    @property
    def symbol(self) -> str:
        return "tan"

    def eval(self, children: list[complex]) -> complex:
        return cmath.tan(children[0])

    def is_float(self) -> bool:
        return self.operand.is_float()


class ArcSin(Function):
    @property
    def symbol(self) -> str:
        return "asin"

    def eval(self, children: list[complex]) -> complex:
        return cmath.asin(children[0])

    def is_float(self) -> bool:
        return self.operand.is_float()


class ArcCos(Function):
    @property
    def symbol(self) -> str:
        return "acos"

    def eval(self, children: list[complex]) -> complex:
        return cmath.acos(children[0])

    def is_float(self) -> bool:
        return self.operand.is_float()


class ArcTan(Function):
    @property
    def symbol(self) -> str:
        return "atan"

    def eval(self, children: list[complex]) -> complex:
        return cmath.atan(children[0])

    def is_float(self) -> bool:
        return self.operand.is_float()


class Sqrt(Function):
    @property
    def symbol(self) -> str:
        return "sqrt"

    def eval(self, children: list[complex]) -> complex:
        return cmath.sqrt(children[0])

    def simplify(self) -> Expression | complex:
        simpl = super().simplify()
        return simplify_product(simpl)


class Exp(Function):
    @property
    def symbol(self) -> str:
        return "exp"

    def eval(self, children: list[complex]) -> complex:
        return cmath.exp(children[0])

    def conjugate(self) -> Expression | complex:
        return Exp(self.operand.conjugate()).simplify()

    def is_float(self) -> bool:
        return self.operand.is_float()


class Log(Function):
    @property
    def symbol(self) -> str:
        return "log"

    def eval(self, children: list[complex]) -> complex:
        return cmath.log(children[0])

    def conjugate(self) -> Expression | complex:
        return Log(self.operand.conjugate()).simplify()

    def is_float(self) -> bool:
        return self.operand.is_float()


class Conjugate(Function):
    @property
    def symbol(self) -> str:
        return "conj"

    def eval(self, children: list[complex]) -> complex:
        return children[0].conjugate()

    def simplify(self) -> Expression | complex:
        simpl = super().simplify()
        if isinstance(simpl, Conjugate):
            return simpl.operand.conjugate()
        return simpl

    def conjugate(self) -> Expression | complex:
        return self.operand


class Associativity(Enum):
    No = 0
    Left = 1
    Right = 2


class Binary(Compound):
    def __init__(self, lhs, rhs):
        super().__init__((lhs, rhs))

    @classmethod
    def from_iterable(cls, operands: typing.Iterable[Expression]):
        it = iter(operands)
        lhs = next(it)
        rhs = next(it)
        return cls(lhs, rhs)

    @property
    def lhs(self) -> Expression:
        return self[0]

    @property
    def rhs(self) -> Expression:
        return self[1]

    @property
    def symbol(self) -> str: ...

    @property
    def precedence(self) -> int: ...

    @property
    def associativity(self) -> Associativity: ...

    def format(self, precedence: int, target: list[str]):
        need_parentheses = precedence > self.precedence
        if need_parentheses:
            target.append("(")
        if self.associativity == Associativity.Left:
            left_precedence = self.precedence
        else:
            left_precedence = self.precedence + 1
        format_expr(self.lhs, left_precedence, target)
        target.append(" ")
        target.append(self.symbol)
        target.append(" ")
        if self.associativity == Associativity.Right:
            right_precedence = self.precedence
        else:
            right_precedence = self.precedence + 1
        format_expr(self.rhs, right_precedence, target)
        if need_parentheses:
            target.append(")")


def add_linear_combination_terms(
    terms: dict[Expression | complex, complex], coef: complex, expr: Expression | complex
) -> None:
    if isinstance(expr, Add):
        add_linear_combination_terms(terms, coef, expr.lhs)
        add_linear_combination_terms(terms, coef, expr.rhs)
    elif isinstance(expr, Sub):
        add_linear_combination_terms(terms, coef, expr.lhs)
        add_linear_combination_terms(terms, -coef, expr.rhs)
    elif isinstance(expr, Mul):
        coef_others = expr.separe_coefficient()
        if coef_others is None:
            terms[expr] = terms.get(expr, 0) + coef
        else:
            mul_coef, others = coef_others
            add_linear_combination_terms(terms, coef * mul_coef, others)
    elif isinstance(expr, Minus):
        add_linear_combination_terms(terms, -coef, expr.operand)
    elif isinstance(expr, complex):
        terms[None] = terms.get(None, 0) + coef * expr
    else:
        terms[expr] = terms.get(expr, 0) + coef


def syntactic_order_key(key: Expression | complex | None) -> Any:
    if key is None:
        return ("value", 0)
    elif isinstance(key, Compound):
        return ("compound", (key.symbol, list(map(syntactic_order_key, key.children))))
    elif isinstance(key, Parameter):
        return ("parameter", key.name)
    elif isinstance(key, complex):
        return ("constant", hash(key))
    else:
        assert False


def syntactic_order(item: tuple[Expression | None, float]) -> Any:
    return syntactic_order_key(item[0])


def rebuild_linear_combination(terms: dict[Expression | None, float]) -> Expression | complex:
    result = complex(0)
    constant_part = terms.get(None, None)
    if constant_part is not None:
        result += constant_part
    for operand, coef in sorted(terms.items(), key=syntactic_order):
        if coef == 0 or operand is None:
            continue
        if coef == 1:
            if result == 0:
                result = operand
            elif isinstance(result, Minus):
                result = Sub(operand, result.operand)
            elif isinstance(result, float) and result < 0:
                result = Sub(operand, -result)
            elif isinstance(result, complex):
                result = Add(result, operand)
            else:
                result = Add(result, operand)
        elif coef == -1:
            if result == 0:
                result = Minus(operand)
            elif isinstance(result, Minus):
                result = Minus(Add(result.operand, operand))
            elif isinstance(result, float) and result < 0:
                result = Minus(Add(operand, -result))
            else:
                result = Sub(result, operand)
        elif coef.imag == 0 and coef.real < 0:
            if result == 0:
                result = Minus(Mul.assoc(-coef, operand))
            elif isinstance(result, Minus):
                result = Minus(Add(result.operand, Mul.assoc(-coef, operand)))
            elif isinstance(result, complex):
                result = Sub(result, Mul.assoc(-coef, operand))
            else:
                result = Sub(result, Mul.assoc(-coef, operand))
        else:
            if result == 0:
                result = Mul.assoc(coef, operand)
            elif isinstance(result, Minus):
                result = Sub(Mul.assoc(coef, operand), result.operand)
            elif isinstance(result, complex):
                result = Add(result, Mul.assoc(coef, operand))
            else:
                result = Add(result, Mul.assoc(coef, operand))
    return result


def decompose_factors(factors: list[Expression], expr: Expression):
    if isinstance(expr, Mul):
        decompose_factors(factors, expr.lhs)
        decompose_factors(factors, expr.rhs)
    else:
        factors.append(expr)


def simplify_linear_combination(expr: Expression | complex) -> Expression | complex:
    if isinstance(expr, complex):
        return expr
    terms = {}
    add_linear_combination_terms(terms, complex(1), expr)
    return rebuild_linear_combination(terms)


class Sum(Binary):
    def simplify(self) -> Expression | complex:
        simpl = super().simplify()
        return simplify_linear_combination(simpl)

    def conjugate(self) -> Expression | complex:
        return simplify_linear_combination(self.__class__(self.lhs.conjugate(), self.rhs.conjugate()))

    def is_float(self) -> bool:
        return lhs.is_float() and rhs.is_float()

    def is_integer(self) -> bool:
        return lhs.is_integer() and rhs.is_integer()


class Add(Sum):
    @property
    def symbol(self) -> str:
        return "+"

    @property
    def precedence(self) -> int:
        return 0

    @property
    def associativity(self) -> Associativity:
        return Associativity.Left

    def eval(self, children: list[complex]) -> complex:
        return children[0] + children[1]


class Sub(Sum):
    @property
    def symbol(self) -> str:
        return "-"

    @property
    def precedence(self) -> int:
        return 0

    @property
    def associativity(self) -> Associativity:
        return Associativity.Left

    def eval(self, children: list[complex]) -> complex:
        return children[0] - children[1]

    def simplify(self) -> Expression | complex:
        simpl = super().simplify()
        return simplify_linear_combination(simpl)


def add_product(products: list[dict[Expression | None, float]], power: float, expr: Expression) -> None:
    if isinstance(expr, Mul):
        add_product(products, power, expr.lhs)
        add_product(products, power, expr.rhs)
    elif isinstance(expr, Div):
        add_product(products, power, expr.lhs)
        add_product(products, -power, expr.rhs)
    elif isinstance(expr, Pow) and is_constant_float(expr.rhs):
        add_product(products, power * expr.rhs.real, expr.lhs)
    elif isinstance(expr, Add) and power.is_integer() and power >= 1:
        for _ in range(int(power)):
            cloned = [factors.copy() for factors in products]
            add_product(products, 1.0, expr.lhs)
            add_product(cloned, 1.0, expr.rhs)
            products.extend(cloned)
    elif isinstance(expr, Sub) and power.is_integer() and power >= 1:
        for _ in range(int(power)):
            cloned = [factors.copy() for factors in products]
            for factors in cloned:
                factors[None] = -factors.get(None, complex(1))
            add_product(products, 1.0, expr.lhs)
            add_product(cloned, 1.0, expr.rhs)
            products.extend(cloned)
    elif isinstance(expr, Minus):
        for factors in products:
            factors[None] = -factors.get(None, complex(1))
        add_product(products, power, expr.operand)
    elif isinstance(expr, complex):
        for factors in products:
            factors[None] = factors.get(None, complex(1)) * expr**power
    else:
        for factors in products:
            factors[expr] = factors.get(expr, 0) + power


def rebuild_product(factors: dict[Expression | None, float]) -> Expression | complex:
    constant = factors.get(None, None)
    if constant is None:
        result = complex(1)
    else:
        result = constant
    for operand, power in sorted(factors.items(), key=syntactic_order):
        if power == 0 or operand is None:
            continue
        if power == 1:
            if result == 1:
                result = operand
            else:
                result = Mul(result, operand)
        elif power == -1:
            result = Div(result, operand)
        elif power > 0:
            if result == 1:
                result = Pow(operand, complex(power))
            else:
                result = Mul(result, Pow(operand, complex(power)))
        else:
            result = Div(result, Pow(operand, complex(-power)))
    return result


def simplify_product(expr: Expression | complex) -> Expression | complex:
    if isinstance(expr, complex):
        return expr
    products = [{}]
    add_product(products, 1.0, expr)
    terms = {}
    for factors in products:
        add_linear_combination_terms(terms, complex(1), rebuild_product(factors))
    return rebuild_linear_combination(terms)


class Product(Binary):
    def simplify(self) -> Expression | complex:
        simpl = super().simplify()
        return simplify_product(simpl)

    def is_float(self) -> bool:
        return lhs.is_float() and rhs.is_float()


class MulDiv(Product):
    @property
    def precedence(self) -> int:
        return 1

    @property
    def associativity(self) -> Associativity:
        return Associativity.Left

    def conjugate(self) -> Expression | complex:
        return simplify_product(self.__class__(self.lhs.conjugate(), self.rhs.conjugate()))


class Mul(MulDiv):
    @property
    def symbol(self) -> str:
        return "*"

    def eval(self, children: list[complex]) -> complex:
        return children[0] * children[1]

    def separe_coefficient(self) -> tuple[complex, Expression] | None:
        current = self
        factors = [current.rhs]
        while not isinstance(current.lhs, complex):
            if isinstance(current.lhs, Mul):
                current = current.lhs
                factors.append(current.rhs)
            else:
                return None
        return (current.lhs, functools.reduce(Mul, reversed(factors)))

    @staticmethod
    def assoc(lhs: Expression, rhs: Expression) -> Expression:
        if isinstance(rhs, Mul):
            return Mul(Mul.assoc(lhs, rhs.lhs), rhs.rhs)
        elif isinstance(rhs, Div):
            return Div(Mul.assoc(lhs, rhs.lhs), rhs.rhs)
        else:
            return Mul(lhs, rhs)

    def is_integer(self) -> bool:
        return lhs.is_integer() and rhs.is_integer()


class Div(MulDiv):
    @property
    def symbol(self) -> str:
        return "/"

    def eval(self, children: list[complex]) -> complex:
        return children[0] / children[1]


class Pow(Product):
    @property
    def symbol(self) -> str:
        return "**"

    @property
    def precedence(self) -> int:
        return 2

    @property
    def associativity(self) -> Associativity:
        return Associativity.Right

    def eval(self, children: list[complex]) -> complex:
        return children[0] ** children[1]

    def simplify(self) -> Expression | complex:
        simpl = super().simplify()
        if isinstance(simpl, Pow) and isinstance(simpl.lhs, Sin) and isinstance(simpl.rhs, complex) and simpl.rhs == 2:
            return 1 - Cos(simpl.lhs.operand) ** 2
        return simpl

    def conjugate(self) -> Expression | complex:
        if is_integer(self.rhs):
            return Pow(self.operand.conjugate(), self.rhs).simplify()
        return super().conjugate()

    def is_integer(self) -> bool:
        return lhs.is_integer() and rhs.is_integer()


def subs(value, variable: Parameter, substitute: Expression | numbers.Number):
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
