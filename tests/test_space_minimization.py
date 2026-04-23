from __future__ import annotations

from graphix import Pattern, command
from graphix.space_minimization import SpaceMinimizationHeuristics


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
    # former heuristics, without `keep_measurement_order_unchanged`
    p.minimize_space(
        [
            SpaceMinimizationHeuristics.causal_flow,
            SpaceMinimizationHeuristics.greedy_degree,
        ]
    )
    after = p.max_space()
    assert after > before


def test_minimization_by_degree_edge_ordering() -> None:
    # Vertices in edges returned by `graph.edges()` are ordered by
    # index: for instance, the edge {1, 2} is represented as `(1, 2)`
    # and not `(2, 1)`.  By contrast, `graph.edges(2)` always returns
    # `2` as the first component, as in `(2, 1)`.
    #
    # Therefore, if we do not treat edges as undirected when computing
    # the degree of node `2` as the intersection of `graph.edges(2)`
    # and `graph.edges()`, we miss the edges `(0, 2)` and `(1, 2)`.
    #
    # The following test constructs an example where node `2` is
    # connected to `0` and `1`, while `0` and `1` are not connected to
    # each other:
    #
    # - If the minimization heuristic determines that `0` and `1` have
    #   degree 1 (which is correct), but that `2` has degree 0 (which
    #   is incorrect), then it selects `2` first, leading to a maximum
    #   space of 3.
    #
    # - By contrast, if `0` and `1` are processed first, because their
    #   degrees are less than that of `2` (which is 2), then the
    #   maximum space is 2.
    #
    # See https://github.com/TeamGraphix/graphix/pull/481#discussion_r3078458069

    p = Pattern(
        cmds=[
            command.N(0),
            command.N(1),
            command.N(2),
            command.E((0, 2)),
            command.E((1, 2)),
            command.M(0),
            command.M(1),
            command.M(2),
        ]
    )
    # Verify Networkx node ordering behaves as expected
    graph = p.extract_graph()
    assert set(graph.edges()) == {(0, 2), (1, 2)}
    assert set(graph.edges(2)) == {(2, 0), (2, 1)}

    p.minimize_space([SpaceMinimizationHeuristics.greedy_degree])
    assert p.max_space() == 2
