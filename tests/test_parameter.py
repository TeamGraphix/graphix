import numpy as np
import pytest

from graphix.parameter import Placeholder
from graphix.pattern import Pattern
from graphix.transpiler import Circuit


def test_placeholder() -> None:
    pattern = Pattern(input_nodes=[0, 1])
    pattern.add(["M", 0, "XY", 0, [], []])
    # A pattern without parameterized angle is not parameterized.
    assert not pattern.is_parameterized()
    # Substitution in a pattern without parameterized angle is the identity.
    alpha = Placeholder("alpha")
    assert list(pattern) == list(pattern.subs(alpha, 0))
    # A pattern without parameterized angle can be simulated.
    pattern.simulate_pattern()
    pattern.add(["M", 1, "XY", alpha, [], []])
    assert pattern.is_parameterized()
    # Parameterized patterns can be substituted, even if some angles are not parameterized.
    pattern0 = pattern.subs(alpha, 0)
    # If all parameterized angles have been instantiated, the pattern is no longer parameterized.
    assert not pattern0.is_parameterized()
    assert list(pattern0) == [["M", 0, "XY", 0, [], []], ["M", 1, "XY", 0, [], []]]
    # Instantied patterns can be simulated.
    pattern0.simulate_pattern()
    pattern1 = pattern.subs(alpha, 1)
    assert not pattern1.is_parameterized()
    assert list(pattern1) == [["M", 0, "XY", 0, [], []], ["M", 1, "XY", 1, [], []]]
    pattern1.simulate_pattern()
    beta = Placeholder("beta")
    pattern.add(["N", 2])
    pattern.add(["M", 2, "XY", beta, [], []])
    # A partially instantiated pattern is still parameterized.
    assert pattern.subs(alpha, 2).is_parameterized()
    pattern23 = pattern.subs(alpha, 2).subs(beta, 3)
    # A full instantiated pattern is no longer parameterized.
    assert not pattern23.is_parameterized()
    assert list(pattern23) == [
        ["M", 0, "XY", 0, [], []],
        ["M", 1, "XY", 2, [], []],
        ["N", 2],
        ["M", 2, "XY", 3, [], []],
    ]
    pattern23.simulate_pattern()
