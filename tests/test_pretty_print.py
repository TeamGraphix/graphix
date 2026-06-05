from __future__ import annotations

import math
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pytest
from numpy.random import PCG64, Generator

from graphix import command, instruction
from graphix.clifford import Clifford
from graphix.fundamentals import ANGLE_PI
from graphix.measurements import Measurement
from graphix.opengraph import OpenGraph
from graphix.parameter import Placeholder
from graphix.pattern import Pattern
from graphix.pretty_print import OutputFormat, complex_to_str, pattern_to_str
from graphix.random_objects import rand_circuit
from graphix.sim.density_matrix import DensityMatrix
from graphix.sim.statevec import Statevec
from graphix.states import BasicStates
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
            command.M(1, Measurement.Y),
            command.M(2, Measurement.XZ(-0.25)),
            command.M(10, Measurement.XZ(-0.25)),
            command.M(3, Measurement.XY(0.1), s_domain={1, 10}, t_domain={2}),
            command.M(4, s_domain={1}, t_domain={2, 3}),
        ]
    )


def test_pattern_repr_example() -> None:
    p = example_pattern()
    assert (
        repr(p)
        == "Pattern(cmds=[N(1), N(2), N(3), N(10), N(4), E((1, 2)), C(1, Clifford.H), M(1, Measurement.Y), M(2, Measurement.XZ(-0.25)), M(10, Measurement.XZ(-0.25)), M(3, Measurement.XY(0.1), {1, 10}, {2}), M(4, s_domain={1}, t_domain={2, 3})])"
    )


def test_pattern_pretty_print_example() -> None:
    p = example_pattern()
    assert (
        str(p)
        == "{2,3}[M(4)]{1} {2}[M(3,pi/10)]{1,10} M(10,XZ,-pi/4) M(2,XZ,-pi/4) M(1,+Y) C(1,H) E(1,2) N(4) N(10) N(3) N(2) N(1)"
    )
    assert p.to_unicode() == "₂₊₃[M₄]¹ ₂[M₃(π/10)]¹⁺¹⁰ M₁₀(XZ,-π/4) M₂(XZ,-π/4) M₁(+Y) C₁(H) E₁₋₂ N₄ N₁₀ N₃ N₂ N₁"
    assert (
        p.to_latex()
        == r"\({}_{2,3}[M_{4}]^{1}\,{}_{2}[M_{3}^{\frac{\pi}{10}}]^{1,10}\,M_{10}^{XZ,-\frac{\pi}{4}}\,M_{2}^{XZ,-\frac{\pi}{4}}\,M_{1}^{+Y}\,C_{1}^{H}\,E_{1,2}\,N_{4}\,N_{10}\,N_{3}\,N_{2}\,N_{1}\)"
    )
    assert (
        pattern_to_str(p, output=OutputFormat.ASCII, limit=9, left_to_right=True)
        == "N(1) N(2) N(3) N(10) N(4) E(1,2) C(1,H) M(1,+Y)...(4 more commands)"
    )


def test_pattern_pretty_print_placeholder() -> None:
    alpha = Placeholder("alpha")
    p = Pattern(input_nodes=[0], cmds=[command.M(0, Measurement.XY(alpha + 0.5))])
    assert str(p) == "M(0,pi*alpha+pi/2)"
    assert p.to_unicode() == "M₀(π×alpha+π/2)"
    assert p.to_latex() == r"\(M_{0}^{\pi \times alpha+\frac{\pi}{2}}\)"


@pytest.mark.parametrize("jumps", range(1, 11))
@pytest.mark.parametrize("output", list(OutputFormat))
def test_pattern_pretty_print_random(fx_bg: PCG64, jumps: int, output: OutputFormat) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    rand_pat = rand_circuit(5, 5, rng=rng).transpile().pattern
    pattern_to_str(rand_pat, output)


@pytest.mark.parametrize("jumps", range(1, 11))
@pytest.mark.parametrize(
    "flow_extractor",
    [
        lambda og: OpenGraph.extract_causal_flow(og.to_bloch()),
        lambda og: OpenGraph.extract_gflow(og.to_bloch()),
        OpenGraph.extract_pauli_flow,
    ],
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
    xzcorr = (
        rand_circuit(5, 5, rng=rng)
        .transpile()
        .pattern.extract_opengraph()
        .to_bloch()
        .extract_causal_flow()
        .to_corrections()
    )

    xzcorr.to_ascii()
    xzcorr.to_latex()
    xzcorr.to_unicode()


def example_og() -> OpenGraph[Measurement]:
    return OpenGraph(
        graph=nx.Graph([(1, 3), (2, 4), (3, 4), (3, 5), (4, 6)]),
        input_nodes=[1, 2],
        output_nodes=[5, 6],
        measurements={
            1: Measurement.XY(0.1),
            2: Measurement.XY(0.2),
            3: Measurement.XY(0.3),
            4: Measurement.XY(0.4),
        },
    )


def test_cflow_str() -> None:
    flow = example_og().to_bloch().extract_causal_flow()

    assert str(flow) == "c(3) = {5}, c(4) = {6}, c(1) = {3}, c(2) = {4}; {1, 2} < {3, 4} < {5, 6}"

    assert (
        flow.to_latex()
        == r"c(3) = \{5\}, \;c(4) = \{6\}, \;c(1) = \{3\}, \;c(2) = \{4\}; \;\{1, 2\} \prec \{3, 4\} \prec \{5, 6\}"
    )

    assert flow.to_unicode() == "c(3) = {5}, c(4) = {6}, c(1) = {3}, c(2) = {4}; {1, 2} ≺ {3, 4} ≺ {5, 6}"

    assert flow.to_ascii(multiline=True) == "c(3) = {5}\nc(4) = {6}\nc(1) = {3}\nc(2) = {4}\n{1, 2} < {3, 4} < {5, 6}"

    assert (
        flow.to_latex(multiline=True)
        == r"c(3) = \{5\},\\c(4) = \{6\},\\c(1) = \{3\},\\c(2) = \{4\};\\\{1, 2\} \prec \{3, 4\} \prec \{5, 6\}"
    )

    assert flow.to_unicode(multiline=True) == "c(3) = {5}\nc(4) = {6}\nc(1) = {3}\nc(2) = {4}\n{1, 2} ≺ {3, 4} ≺ {5, 6}"


def test_gflow_str() -> None:
    flow = example_og().to_bloch().extract_gflow()

    assert str(flow) == "g(1) = {3, 6}, g(2) = {4, 5}, g(3) = {5}, g(4) = {6}; {1, 2} < {3, 4} < {5, 6}"


def test_pflow_str() -> None:
    flow = example_og().extract_pauli_flow()

    assert str(flow) == "p(1) = {3, 6}, p(2) = {4, 5}, p(3) = {5}, p(4) = {6}; {1, 2} < {3, 4} < {5, 6}"


def test_xzcorr_str() -> None:
    flow = example_og().to_bloch().extract_causal_flow().to_corrections()

    assert (
        str(flow)
        == "x(3) = {5}, x(4) = {6}, x(1) = {3}, x(2) = {4}; z(1) = {4, 5}, z(2) = {3, 6}; {1, 2} < {3, 4} < {5, 6}"
    )


def test_complex_to_str_issue_examples() -> None:
    # The three canonical examples from the issue.
    assert complex_to_str(0.25, OutputFormat.ASCII) == "1/4"
    assert complex_to_str(2**-0.5, OutputFormat.Unicode) == "√2/2"
    assert complex_to_str(0.5 + math.sqrt(3) / 2 * 1j, OutputFormat.Unicode) == "e^(iπ/3)"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "0"),
        (1e-12, "0"),
        (1, "1"),
        (-1, "-1"),
        (2, "2"),
        (0.5, "1/2"),
        (-0.25, "-1/4"),
        (2**-0.5, "√2/2"),
        (math.sqrt(3) / 2, "√3/2"),
        (1j, "i"),
        (-1j, "-i"),
        (0.5j, "1/2i"),
        (-(2**-0.5) * 1j, "-√2/2i"),
    ],
)
def test_complex_to_str_unicode_values(value: complex, expected: str) -> None:
    assert complex_to_str(value, OutputFormat.Unicode) == expected


def test_complex_to_str_exponentials() -> None:
    assert complex_to_str(1j, OutputFormat.Unicode) == "i"
    assert complex_to_str(math.cos(math.pi / 4) + math.sin(math.pi / 4) * 1j, OutputFormat.Unicode) == "e^(iπ/4)"
    # Negative phase keeps the sign inside the exponent.
    assert complex_to_str(math.cos(math.pi / 3) - math.sin(math.pi / 3) * 1j, OutputFormat.Unicode) == "e^(-iπ/3)"
    assert complex_to_str(0.5 + math.sqrt(3) / 2 * 1j, OutputFormat.ASCII) == "e^(i*pi/3)"


def test_complex_to_str_latex() -> None:
    assert complex_to_str(2**-0.5, OutputFormat.LaTeX) == r"\frac{\sqrt{2}}{2}"
    assert complex_to_str(0.25, OutputFormat.LaTeX) == r"\frac{1}{4}"
    assert complex_to_str(0.5 + math.sqrt(3) / 2 * 1j, OutputFormat.LaTeX) == r"\mathrm{e}^{\mathrm{i} \frac{\pi}{3}}"


def test_complex_to_str_fallback_and_symbolic() -> None:
    # An unrecognized value falls back to a rounded decimal.
    assert complex_to_str(0.123456, OutputFormat.ASCII) == "0.1235"
    # A non-numeric object is stringified rather than raising.
    assert complex_to_str("alpha", OutputFormat.ASCII) == "alpha"


def test_statevec_draw() -> None:
    bell = Statevec([2**-0.5, 0, 0, 2**-0.5])
    assert bell.draw(OutputFormat.Unicode) == "√2/2|00⟩ + √2/2|11⟩"
    assert bell.draw(OutputFormat.ASCII) == "sqrt(2)/2|00> + sqrt(2)/2|11>"


def test_statevec_draw_single_basis_state() -> None:
    state = Statevec(data=[BasicStates.ZERO, BasicStates.ONE])
    assert state.draw(OutputFormat.Unicode) == "|01⟩"
    # LSB encoding reverses the ket label.
    assert state.draw(OutputFormat.Unicode, encoding="LSB") == "|10⟩"


def test_density_matrix_draw() -> None:
    dm = DensityMatrix(data=[BasicStates.ZERO])
    assert dm.draw(OutputFormat.ASCII) == "[ 1  0 ]\n[ 0  0 ]"
    assert dm.draw(OutputFormat.LaTeX) == r"\begin{pmatrix}1 & 0 \\ 0 & 0\end{pmatrix}"


def test_complex_to_str_exponential_with_radius() -> None:
    # |z| != 1: the radius prefixes the exponential form (1 + i = √2 e^{iπ/4}).
    assert complex_to_str(1 + 1j, OutputFormat.Unicode) == "√2·e^(iπ/4)"
    assert complex_to_str(1 + 1j, OutputFormat.ASCII) == "sqrt(2)*e^(i*pi/4)"
    assert complex_to_str(1 + 1j, OutputFormat.LaTeX) == r"\sqrt{2} \mathrm{e}^{\mathrm{i} \frac{\pi}{4}}"


def test_complex_to_str_cartesian_form() -> None:
    # Both parts are recognized but the phase is not a simple fraction of π, so the
    # cartesian form is used instead of the exponential one.
    assert complex_to_str(0.5 + 0.25j, OutputFormat.Unicode) == "1/2 + 1/4i"
    assert complex_to_str(0.5 + 0.25j, OutputFormat.LaTeX) == r"\frac{1}{2} + \frac{1}{4}\mathrm{i}"


def test_complex_to_str_complex_decimal_fallback() -> None:
    # Neither part is a recognized value -> rounded decimal real and imaginary parts.
    assert complex_to_str(0.123456 + 0.234567j, OutputFormat.Unicode) == "0.1235+0.2346i"


def test_complex_to_str_imaginary_formats() -> None:
    assert complex_to_str(0.5j, OutputFormat.LaTeX) == r"\frac{1}{2}\mathrm{i}"
    assert complex_to_str(0.5j, OutputFormat.ASCII) == "1/2i"


def test_complex_to_str_integer_times_sqrt() -> None:
    assert complex_to_str(math.sqrt(12), OutputFormat.Unicode) == "2√3"


def test_statevec_draw_negative_and_parenthesized() -> None:
    # Negative amplitudes use a `-` separator between terms.
    neg = Statevec([0.5, -0.5, 0.5, 0.5])
    assert neg.draw(OutputFormat.Unicode) == "1/2|00⟩ - 1/2|01⟩ + 1/2|10⟩ + 1/2|11⟩"
    # A compound (cartesian) amplitude is parenthesized before the ket. Build from a
    # numpy array so the amplitudes are ``numpy.complex128`` (Python's ``complex`` only
    # gained ``__complex__`` in 3.11, so a bare ``complex`` is rejected on 3.10).
    binomial = Statevec(np.array([0.5 + 0.25j, (1 - abs(0.5 + 0.25j) ** 2) ** 0.5]))
    assert binomial.draw(OutputFormat.Unicode) == "(1/2 + 1/4i)|0⟩ + √11/4|1⟩"
    # A unit negative amplitude collapses to a bare `-|ket⟩`.
    assert Statevec([-1.0, 0.0]).draw(OutputFormat.Unicode) == "-|0⟩"


def test_complex_to_str_precision_is_configurable() -> None:
    z = 0.123456 + 0.234567j
    assert complex_to_str(z, OutputFormat.ASCII, precision=2) == "0.12+0.23i"
    assert complex_to_str(z, OutputFormat.ASCII, precision=6) == "0.123456+0.234567i"
    # The default keeps the previous behaviour (four significant digits).
    assert complex_to_str(z, OutputFormat.ASCII) == "0.1235+0.2346i"
