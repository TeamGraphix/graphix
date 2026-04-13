from __future__ import annotations

from graphix import Pattern, command
from graphix.space_minimization import greedy_minimization_by_degree, minimization_using_causal_flow


def counter_example_issue_454(sz: int, depth: int) -> Pattern:
    block = Pattern(input_nodes=range(sz))
    block.extend(command.N(sz + i) for i in range(sz))
    for i in range(1, sz):
        block.extend(command.E((j, sz + i)) for j in range(sz))
        block.extend(command.E((sz + j, sz + i)) for j in range(1, i))
    block.extend(command.M(i) for i in range(sz))
    p, _ = block.compose(block, dict(zip(block.input_nodes, block.output_nodes, strict=True)))
    for _ in range(depth):
        p, _ = p.compose(block, dict(zip(block.input_nodes, p.output_nodes, strict=True)))
    return p


def test_minimize_space() -> None:
    p = counter_example_issue_454(sz=4, depth=3)
    before = p.max_space()
    p.minimize_space()
    after = p.max_space()
    assert after <= before


def test_minimize_space_deprecated() -> None:
    p = counter_example_issue_454(sz=4, depth=3)
    before = p.max_space()
    # former heuristics, without `do_nothing_for_space_minimization`
    p.minimize_space([minimization_using_causal_flow, greedy_minimization_by_degree])
    after = p.max_space()
    assert after > before
