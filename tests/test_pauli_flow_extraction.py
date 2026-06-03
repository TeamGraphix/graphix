r"""Tests for Pauli-flow extraction from a pattern / XZ-corrections.

Correctness criterion
---------------------
A reconstructed Pauli flow ``pf`` generates the original pattern if and only if
``pf.check_well_formed()`` succeeds *and* ``pf.to_corrections()`` reproduces the pattern's
X- and Z-corrections exactly. The latter "round-trip" property is the decisive check: it
guarantees that the flow generates *this* pattern (and not merely some Pauli flow of the
underlying open graph, which need not be unique). The tests below verify this on the three
worked examples of the issue, on a Pauli-measured open graph, and on a randomized family of
open graphs that admit a Pauli flow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pytest

from graphix import Measurement, OpenGraph, Pattern
from graphix.command import E, M, N, X, Z
from graphix.flow.core import _solve_gf2

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from collections.abc import Set as AbstractSet

    from numpy.random import Generator


def _norm(corrections: Mapping[int, AbstractSet[int]]) -> dict[int, frozenset[int]]:
    """Drop empty correction sets to compare correction dictionaries up to empty entries."""
    return {k: frozenset(v) for k, v in corrections.items() if v}


def _assert_round_trip(pattern: Pattern) -> None:
    xz = pattern.extract_xzcorrections()
    pf = xz.to_pauli_flow()  # `check_well_formed` runs inside `to_pauli_flow`.
    assert pf.is_well_formed()
    rt = pf.to_corrections()
    assert _norm(rt.x_corrections) == _norm(xz.x_corrections)
    assert _norm(rt.z_corrections) == _norm(xz.z_corrections)


def _correction_function(pattern: Pattern) -> dict[int, set[int]]:
    pf = pattern.extract_pauli_flow()
    return {k: set(v) for k, v in pf.correction_function.items()}


def _causal_pattern() -> Pattern:
    return Pattern(input_nodes=[0], cmds=[N(1), E((0, 1)), M(0, Measurement.XY(0)), X(1, {0})], output_nodes=[1])


def _gflow_pattern() -> Pattern:
    return Pattern(
        input_nodes=[0],
        cmds=[
            N(1), N(2), N(3), E((0, 1)), E((0, 2)), E((1, 2)), E((1, 3)),
            M(0, Measurement.XY(0.1)), X(2, {0}), X(3, {0}),
            M(1, Measurement.XZ(0.2)), Z(2, {1}), Z(3, {1}), X(2, {1}),
        ],
        output_nodes=[2, 3],
    )  # fmt: skip


def _pauli_pattern() -> Pattern:
    return Pattern(
        input_nodes=[0],
        cmds=[
            N(1), N(2), N(3), E((0, 1)), E((1, 2)), E((2, 3)),
            M(0, Measurement.X), X(3, {0}),
            M(1, Measurement.X), Z(3, {1}),
            M(2, Measurement.X), X(3, {2}),
        ],
        output_nodes=[3],
    )  # fmt: skip


def test_extract_pauli_flow_causal_example() -> None:
    pattern = _causal_pattern()
    assert _correction_function(pattern) == {0: {1}}
    _assert_round_trip(pattern)


def test_extract_pauli_flow_gflow_example() -> None:
    pattern = _gflow_pattern()
    assert _correction_function(pattern) == {0: {2, 3}, 1: {1, 2}}
    _assert_round_trip(pattern)


def test_extract_pauli_flow_pauli_example() -> None:
    # The flow must include the anachronical correction (node 1 in p(0)) that does not
    # appear in the pattern, in order to satisfy the X-axis proposition (P7).
    pattern = _pauli_pattern()
    assert _correction_function(pattern) == {0: {1, 3}, 1: {2}, 2: {3}}
    _assert_round_trip(pattern)


@pytest.mark.filterwarnings("ignore:Open graph with non-inferred Pauli measurements.")
def test_extract_pauli_flow_pauli_opengraph() -> None:
    og = OpenGraph(
        graph=nx.Graph([(0, 2), (2, 4), (3, 4), (4, 6), (1, 4), (1, 6), (2, 3), (3, 5), (2, 6), (3, 6)]),
        input_nodes=[0],
        output_nodes=[5, 6],
        measurements={
            0: Measurement.XY(0.1),
            1: Measurement.XZ(0.1),
            2: Measurement.Y,
            3: Measurement.XY(0.1),
            4: Measurement.Z,
        },
    )
    _assert_round_trip(og.to_pattern())


_MEASUREMENTS: list[Callable[[Generator], Measurement]] = [
    lambda r: Measurement.XY(round(float(r.random()), 3)),
    lambda r: Measurement.XZ(round(float(r.random()), 3)),
    lambda r: Measurement.YZ(round(float(r.random()), 3)),
    lambda _r: Measurement.X,
    lambda _r: Measurement.Y,
    lambda _r: Measurement.Z,
]


@pytest.mark.filterwarnings("ignore:Open graph with non-inferred Pauli measurements.")
def test_extract_pauli_flow_randomized_round_trip() -> None:
    # Generate random open graphs; those that admit a Pauli flow (so that `to_pattern`
    # succeeds) are converted to a pattern, and the reconstructed flow is checked to be
    # well formed and to reproduce the pattern's corrections.
    tested = 0
    for seed in range(400):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(4, 10))
        graph = nx.gnp_random_graph(n, 0.45, seed=seed)
        if graph.number_of_edges() == 0:
            continue
        nodes = list(graph.nodes())
        rng.shuffle(nodes)
        n_out = int(rng.integers(1, max(2, n // 2)))
        n_in = int(rng.integers(0, max(1, n // 2)))
        outputs = nodes[:n_out]
        inputs = nodes[n_out : n_out + n_in]
        measurements = {
            m: _MEASUREMENTS[int(rng.integers(0, len(_MEASUREMENTS)))](rng) for m in nodes if m not in outputs
        }
        try:
            pattern = OpenGraph(
                graph=graph, input_nodes=inputs, output_nodes=outputs, measurements=measurements
            ).to_pattern()
        except Exception:  # noqa: BLE001, S112  open graph without a flow -> not a valid test case
            continue
        _assert_round_trip(pattern)
        tested += 1
    assert tested >= 30  # ensure the randomized sweep actually exercised the extraction


def test_solve_gf2_unique_solution() -> None:
    # x0 + x1 = 1, x1 = 1  ->  x0 = 0, x1 = 1
    assert _solve_gf2([[1, 1], [0, 1]], [1, 1], 2) == [0, 1]


def test_solve_gf2_free_variable_set_to_zero() -> None:
    # x0 + x1 = 0  ->  free x1 = 0, x0 = 0
    assert _solve_gf2([[1, 1]], [0], 2) == [0, 0]


def test_solve_gf2_inconsistent_returns_none() -> None:
    # x0 = 0 and x0 = 1 simultaneously.
    assert _solve_gf2([[1, 0], [1, 0]], [0, 1], 2) is None


def test_solve_gf2_no_equations() -> None:
    assert _solve_gf2([], [], 3) == [0, 0, 0]
