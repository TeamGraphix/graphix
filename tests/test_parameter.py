import graphix.command
from graphix.parameter import Placeholder
from graphix.pattern import Pattern


def test_placeholder() -> None:
    pattern = Pattern(input_nodes=[0, 1])
    pattern.add(graphix.command.M(node=0))
    # A pattern without parameterized angle is not parameterized.
    assert not pattern.is_parameterized()
    # Substitution in a pattern without parameterized angle is the identity.
    alpha = Placeholder("alpha")
    assert list(pattern) == list(pattern.subs(alpha, 0))
    # A pattern without parameterized angle can be simulated.
    pattern.simulate_pattern()
    pattern.add(graphix.command.M(node=1, angle=alpha))
    assert pattern.is_parameterized()
    # Parameterized patterns can be substituted, even if some angles are not parameterized.
    pattern0 = pattern.subs(alpha, 0)
    # If all parameterized angles have been instantiated, the pattern is no longer parameterized.
    assert not pattern0.is_parameterized()
    assert list(pattern0) == [graphix.command.M(node=0), graphix.command.M(node=1)]
    # Instantied patterns can be simulated.
    pattern0.simulate_pattern()
    pattern1 = pattern.subs(alpha, 1)
    assert not pattern1.is_parameterized()
    assert list(pattern1) == [graphix.command.M(node=0), graphix.command.M(node=1, angle=1)]
    pattern1.simulate_pattern()
    beta = Placeholder("beta")
    pattern.add(graphix.command.N(node=2))
    pattern.add(graphix.command.M(node=2, angle=beta))
    # A partially instantiated pattern is still parameterized.
    assert pattern.subs(alpha, 2).is_parameterized()
    pattern23 = pattern.subs(alpha, 2).subs(beta, 3)
    # A full instantiated pattern is no longer parameterized.
    assert not pattern23.is_parameterized()
    assert list(pattern23) == [
        graphix.command.M(node=0),
        graphix.command.M(node=1, angle=2),
        graphix.command.N(node=2),
        graphix.command.M(node=2, angle=3),
    ]
    pattern23.simulate_pattern()
