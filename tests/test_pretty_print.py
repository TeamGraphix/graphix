from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import pytest
from numpy.random import PCG64, Generator

from graphix import command, instruction
from graphix.clifford import Clifford
from graphix.fundamentals import ANGLE_PI, Plane
from graphix.measurements import Measurement
from graphix.opengraph import OpenGraph
from graphix.pattern import Pattern
from graphix.pretty_print import OutputFormat, pattern_to_str
from graphix.random_objects import rand_circuit
from graphix.transpiler import Circuit

if TYPE_CHECKING:
    from collections.abc import Callable

    from graphix.flow.core import PauliFlow


def test_circuit_repr() -> None:
    circuit = Circuit(width=3, instr=[instruction.H(0), instruction.RX(1, ANGLE_PI), instruction.CCX(0, (1, 2))])
    assert repr(circuit) == "Circuit(width=3, instr=[H(0), RX(1, pi), CCX(0, (1, 2))])"


def j_alpha() -> Pattern:
    return Pattern(input_nodes=[1], cmds=[command.N(2), command.E((1, 2)), command.M(1), command.X(2, domain={1})])


def test_pattern_repr_j_alpha() -> None:
    p = j_alpha()
    assert repr(p) == "Pattern(input_nodes=[1], cmds=[N(2), E((1, 2)), M(1), X(2, {1})], output_nodes=[2])"


def test_pattern_pretty_print_j_alpha() -> None:
    p = j_alpha()
    assert str(p) == "X(2,{1}) M(1) E(1,2) N(2)"
    assert p.to_unicode() == "X₂¹ M₁ E₁₋₂ N₂"
    assert p.to_latex() == r"\(X_{2}^{1}\,M_{1}\,E_{1,2}\,N_{2}\)"


def example_pattern() -> Pattern:
    return Pattern(
        cmds=[
            command.N(1),
            command.N(2),
            command.N(3),
            command.N(10),
            command.N(4),
            command.E((1, 2)),
            command.C(1, Clifford.H),
            command.M(1, Plane.YZ, 0.5),
            command.M(2, Plane.XZ, -0.25),
            command.M(10, Plane.XZ, -0.25),
            command.M(3, Plane.XY, 0.1, s_domain={1, 10}, t_domain={2}),
            command.M(4, s_domain={1}, t_domain={2, 3}),
        ]
    )


def test_pattern_repr_example() -> None:
    p = example_pattern()
    assert (
        repr(p)
        == "Pattern(cmds=[N(1), N(2), N(3), N(10), N(4), E((1, 2)), C(1, Clifford.H), M(1, Plane.YZ, 0.5), M(2, Plane.XZ, -0.25), M(10, Plane.XZ, -0.25), M(3, angle=0.1, s_domain={1, 10}, t_domain={2}), M(4, s_domain={1}, t_domain={2, 3})])"
    )


def test_pattern_pretty_print_example() -> None:
    p = example_pattern()
    assert (
        str(p)
        == "{2,3}[M(4)]{1} {2}[M(3,pi/10)]{1,10} M(10,XZ,-pi/4) M(2,XZ,-pi/4) M(1,YZ,pi/2) C(1,H) E(1,2) N(4) N(10) N(3) N(2) N(1)"
    )
    assert p.to_unicode() == "₂₊₃[M₄]¹ ₂[M₃(π/10)]¹⁺¹⁰ M₁₀(XZ,-π/4) M₂(XZ,-π/4) M₁(YZ,π/2) C₁(H) E₁₋₂ N₄ N₁₀ N₃ N₂ N₁"
    assert (
        p.to_latex()
        == r"\({}_{2,3}[M_{4}]^{1}\,{}_{2}[M_{3}^{\frac{\pi}{10}}]^{1,10}\,M_{10}^{XZ,-\frac{\pi}{4}}\,M_{2}^{XZ,-\frac{\pi}{4}}\,M_{1}^{YZ,\frac{\pi}{2}}\,C_{1}^{H}\,E_{1,2}\,N_{4}\,N_{10}\,N_{3}\,N_{2}\,N_{1}\)"
    )
    assert (
        pattern_to_str(p, output=OutputFormat.ASCII, limit=9, left_to_right=True)
        == "N(1) N(2) N(3) N(10) N(4) E(1,2) C(1,H) M(1,YZ,pi/2)...(4 more commands)"
    )


@pytest.mark.parametrize("jumps", range(1, 11))
@pytest.mark.parametrize("output", list(OutputFormat))
def test_pattern_pretty_print_random(fx_bg: PCG64, jumps: int, output: OutputFormat) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    rand_pat = rand_circuit(5, 5, rng=rng).transpile().pattern
    pattern_to_str(rand_pat, output)


@pytest.mark.parametrize("jumps", range(1, 11))
@pytest.mark.parametrize(
    "flow_extractor", [OpenGraph.extract_causal_flow, OpenGraph.extract_gflow, OpenGraph.extract_pauli_flow]
)
def test_flow_pretty_print_random(
    fx_bg: PCG64,
    jumps: int,
    flow_extractor: Callable[[OpenGraph[Measurement]], PauliFlow[Measurement]],
) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    rand_og = rand_circuit(5, 5, rng=rng).transpile().pattern.extract_opengraph()
    flow = flow_extractor(rand_og)

    flow.to_ascii()
    flow.to_latex()
    flow.to_unicode()


@pytest.mark.parametrize("jumps", range(1, 11))
def test_xzcorr_pretty_print_random(
    fx_bg: PCG64,
    jumps: int,
) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    xzcorr = rand_circuit(5, 5, rng=rng).transpile().pattern.extract_opengraph().extract_causal_flow().to_corrections()

    xzcorr.to_ascii()
    xzcorr.to_latex()
    xzcorr.to_unicode()


def example_og() -> OpenGraph[Measurement]:
    return OpenGraph(
        graph=nx.Graph([(1, 3), (2, 4), (3, 4), (3, 5), (4, 6)]),
        input_nodes=[1, 2],
        output_nodes=[5, 6],
        measurements={
            1: Measurement(0.1, Plane.XY),
            2: Measurement(0.2, Plane.XY),
            3: Measurement(0.3, Plane.XY),
            4: Measurement(0.4, Plane.XY),
        },
    )


def test_cflow_str() -> None:
    flow = example_og().extract_causal_flow()

    assert str(flow) == "c(3) = {5}, c(4) = {6}, c(1) = {3}, c(2) = {4}\n{1, 2} < {3, 4} < {5, 6}"


def test_gflow_str() -> None:
    flow = example_og().extract_gflow()

    assert str(flow) == "g(1) = {3, 6}, g(2) = {4, 5}, g(3) = {5}, g(4) = {6}\n{1, 2} < {3, 4} < {5, 6}"


def test_pflow_str() -> None:
    flow = example_og().extract_pauli_flow()

    assert str(flow) == "p(1) = {3, 6}, p(2) = {4, 5}, p(3) = {5}, p(4) = {6}\n{1, 2} < {3, 4} < {5, 6}"


def test_xzcorr_str() -> None:
    flow = example_og().extract_causal_flow().to_corrections()

    assert (
        str(flow)
        == "x(3) = {5}, x(4) = {6}, x(1) = {3}, x(2) = {4}\nz(1) = {4, 5}, z(2) = {3, 6}\n{1, 2} < {3, 4} < {5, 6}"
    )
