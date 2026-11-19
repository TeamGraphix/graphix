from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

from graphix import Pattern, command
from graphix.measurements import Measurement
from graphix.opengraph import OpenGraph
from graphix.optimization import StandardizedPattern
from graphix.space_minimization import SpaceMinimizationHeuristics

if TYPE_CHECKING:
    from numpy.random import Generator


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
            SpaceMinimizationHeuristics.extended_causal_flow,
            SpaceMinimizationHeuristics.greedy_degree,
        ]
    )
    after = p.max_space()
    assert after > before


def test_minimize_space_extended_causal_flow(fx_rng: Generator) -> None:
    # [0]-1-2-(3) with a YZ-measured node 4 attached to 1: extended causal flow but no causal flow.
    og = OpenGraph(
        graph=nx.Graph([(0, 1), (1, 2), (2, 3), (1, 4)]),
        input_nodes=[0],
        output_nodes=[3],
        measurements={0: Measurement.XY(0.1), 1: Measurement.XY(0.2), 2: Measurement.XY(0.3), 4: Measurement.YZ(0.4)},
    )
    assert og.to_causalflow_or_none() is None
    p = og.to_extendedcausalflow().to_xzcorrections().to_pattern()

    result = SpaceMinimizationHeuristics.extended_causal_flow(StandardizedPattern.from_pattern(p))
    assert result is not None
    assert result.known_optimal
    assert result.new_order == (0, 4, 1, 2)

    state_ref = p.simulate(rng=fx_rng)
    before = p.max_space()
    p.minimize_space([SpaceMinimizationHeuristics.extended_causal_flow])
    assert p.max_space() < before
    state = p.simulate(rng=fx_rng)
    assert state.isclose(state_ref)


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
    graph = p.to_opengraph().graph
    assert set(graph.edges()) == {(0, 2), (1, 2)}
    assert set(graph.edges(2)) == {(2, 0), (2, 1)}

    p.minimize_space([SpaceMinimizationHeuristics.greedy_degree])
    assert p.max_space() == 2
