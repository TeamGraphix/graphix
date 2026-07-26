from __future__ import annotations

import cmath
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
from graphix.pretty_print import OutputFormat, _factor_uniform_magnitude, complex_to_str, pattern_to_str
from graphix.random_objects import rand_circuit
from graphix.sim.density_matrix import DensityMatrix
from graphix.sim.statevec import Statevector
from graphix.states import BasicStates
from graphix.transpiler import Circuit

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

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
        lambda og: OpenGraph.to_causalflow(og.to_bloch()),
        lambda og: OpenGraph.to_gflow(og.to_bloch()),
        OpenGraph.to_pauliflow,
    ],
)
def test_flow_pretty_print_random(
    fx_bg: PCG64,
    jumps: int,
    flow_extractor: Callable[[OpenGraph[Measurement]], PauliFlow[Measurement]],
) -> None:
    rng = Generator(fx_bg.jumped(jumps))
    rand_og = rand_circuit(5, 5, rng=rng).transpile().pattern.infer_pauli_measurements().to_opengraph()
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
        rand_circuit(5, 5, rng=rng).transpile().pattern.to_opengraph().to_bloch().to_causalflow().to_xzcorrections()
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
    flow = example_og().to_bloch().to_causalflow()

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
    flow = example_og().to_bloch().to_gflow()

    assert str(flow) == "g(1) = {3, 6}, g(2) = {4, 5}, g(3) = {5}, g(4) = {6}; {1, 2} < {3, 4} < {5, 6}"


def test_pflow_str() -> None:
    flow = example_og().to_pauliflow()

    assert str(flow) == "p(1) = {3, 6}, p(2) = {4, 5}, p(3) = {5}, p(4) = {6}; {1, 2} < {3, 4} < {5, 6}"


def test_xzcorr_str() -> None:
    flow = example_og().to_bloch().to_causalflow().to_xzcorrections()

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
        (0, {OutputFormat.Unicode: "0"}),
        (1e-12, {OutputFormat.Unicode: "0"}),
        (1, {OutputFormat.Unicode: "1"}),
        (-1, {OutputFormat.Unicode: "-1"}),
        (2, {OutputFormat.Unicode: "2"}),
        (0.5, {OutputFormat.Unicode: "1/2"}),
        (0.25, {OutputFormat.LaTeX: r"\frac{1}{4}"}),
        (-0.25, {OutputFormat.Unicode: "-1/4"}),
        (2**-0.5, {OutputFormat.Unicode: "√2/2", OutputFormat.LaTeX: r"\frac{\sqrt{2}}{2}"}),
        (math.sqrt(3) / 2, {OutputFormat.Unicode: "√3/2"}),
        (1j, {OutputFormat.Unicode: "i"}),
        (-1j, {OutputFormat.Unicode: "-i"}),
        # The imaginary unit leads the numerator (i/2, not 1/2i).
        (0.5j, {OutputFormat.Unicode: "i/2", OutputFormat.ASCII: "i/2", OutputFormat.LaTeX: r"\frac{\mathrm{i}}{2}"}),
        (-(2**-0.5) * 1j, {OutputFormat.Unicode: "-i√2/2"}),
        # Complex exponentials on the unit circle.
        (math.cos(math.pi / 4) + math.sin(math.pi / 4) * 1j, {OutputFormat.Unicode: "e^(iπ/4)"}),
        # Negative phase keeps the sign inside the exponent.
        (math.cos(math.pi / 3) - math.sin(math.pi / 3) * 1j, {OutputFormat.Unicode: "e^(-iπ/3)"}),
        (
            0.5 + math.sqrt(3) / 2 * 1j,
            {
                OutputFormat.Unicode: "e^(iπ/3)",
                OutputFormat.ASCII: "e^(i*pi/3)",
                OutputFormat.LaTeX: r"\mathrm{e}^{\mathrm{i} \frac{\pi}{3}}",
            },
        ),
        # An unrecognized value falls back to a rounded decimal.
        (0.123456, {OutputFormat.ASCII: "0.1235"}),
        # A non-numeric object is stringified rather than raising.
        ("alpha", {OutputFormat.ASCII: "alpha"}),
        # |z| != 1 with nice Cartesian parts: the Cartesian form is preferred over the
        # radius-prefixed exponential (1 + i rather than √2·e^(iπ/4)).
        (1 + 1j, {OutputFormat.Unicode: "1 + i", OutputFormat.ASCII: "1 + i", OutputFormat.LaTeX: r"1 + \mathrm{i}"}),
        # When the Cartesian parts are not recognized, the radius-prefixed exponential is the
        # last resort before the decimal fallback.
        (
            2 * (math.cos(math.pi / 5) + math.sin(math.pi / 5) * 1j),
            {
                OutputFormat.Unicode: "2·e^(iπ/5)",
                OutputFormat.ASCII: "2*e^(i*pi/5)",
                OutputFormat.LaTeX: r"2 \mathrm{e}^{\mathrm{i} \frac{\pi}{5}}",
            },
        ),
        # Both parts recognized but the phase is not a simple fraction of π: Cartesian form.
        (
            0.5 + 0.25j,
            {OutputFormat.Unicode: "1/2 + 1/4i", OutputFormat.LaTeX: r"\frac{1}{2} + \frac{1}{4}\mathrm{i}"},
        ),
        # Neither part recognized -> rounded decimal real and imaginary parts.
        (0.123456 + 0.234567j, {OutputFormat.Unicode: "0.1235+0.2346i"}),
        # Integer multiple of a surd.
        (math.sqrt(12), {OutputFormat.Unicode: "2√3"}),
    ],
)
def test_complex_to_str_values(value: object, expected: Mapping[OutputFormat, str]) -> None:
    for output, text in expected.items():
        assert complex_to_str(value, output) == text


def test_statevec_draw() -> None:
    bell = Statevector([2**-0.5, 0, 0, 2**-0.5])
    # A magnitude shared by every amplitude is factored out.
    assert bell.draw(OutputFormat.Unicode) == "√2/2(|00⟩ + |11⟩)"
    assert bell.draw(OutputFormat.ASCII) == "sqrt(2)/2(|00> + |11>)"
    # LaTeX uses the \ket{...} macro for the basis kets.
    assert bell.draw(OutputFormat.LaTeX) == r"\frac{\sqrt{2}}{2}(\ket{00} + \ket{11})"


def test_statevec_draw_single_basis_state() -> None:
    state = Statevector(data=[BasicStates.ZERO, BasicStates.ONE])
    assert state.draw(OutputFormat.Unicode) == "|01⟩"
    # LaTeX ket notation for a bare basis state.
    assert state.draw(OutputFormat.LaTeX) == r"\ket{01}"
    # LSB encoding reverses the ket label.
    assert state.draw(OutputFormat.Unicode, encoding="LSB") == "|10⟩"


def test_density_matrix_draw() -> None:
    dm = DensityMatrix(data=[BasicStates.ZERO])
    assert dm.draw(OutputFormat.ASCII) == "[ 1  0 ]\n[ 0  0 ]"
    assert dm.draw(OutputFormat.LaTeX) == r"\begin{pmatrix}1 & 0 \\ 0 & 0\end{pmatrix}"


def test_statevec_draw_negative_and_parenthesized() -> None:
    # The shared 1/2 magnitude is factored out, with the signs kept inside the parentheses.
    neg = Statevector([0.5, -0.5, 0.5, 0.5])
    assert neg.draw(OutputFormat.Unicode) == "1/2(|00⟩ - |01⟩ + |10⟩ + |11⟩)"
    # A compound (cartesian) amplitude is parenthesized before the ket. Build from a
    # numpy array so the amplitudes are ``numpy.complex128`` (Python's ``complex`` only
    # gained ``__complex__`` in 3.11, so a bare ``complex`` is rejected on 3.10).
    binomial = Statevector(np.array([0.5 + 0.25j, (1 - abs(0.5 + 0.25j) ** 2) ** 0.5]))
    assert binomial.draw(OutputFormat.Unicode) == "(1/2 + 1/4i)|0⟩ + √11/4|1⟩"
    # A unit negative amplitude collapses to a bare `-|ket⟩`.
    assert Statevector([-1.0, 0.0]).draw(OutputFormat.Unicode) == "-|0⟩"


def test_complex_to_str_precision_is_configurable() -> None:
    z = 0.123456 + 0.234567j
    assert complex_to_str(z, OutputFormat.ASCII, precision=2) == "0.12+0.23i"
    assert complex_to_str(z, OutputFormat.ASCII, precision=6) == "0.123456+0.234567i"
    # The default keeps the previous behaviour (four significant digits).
    assert complex_to_str(z, OutputFormat.ASCII) == "0.1235+0.2346i"


def test_complex_to_str_rtol_controls_recognition() -> None:
    # A value slightly off 1/2: with the default (tight) tolerances it is not recognized as a
    # fraction and falls back to a decimal; a looser relative tolerance recognizes it as 1/2.
    x = 0.500001
    assert complex_to_str(x, OutputFormat.Unicode) == "0.5"
    assert complex_to_str(x, OutputFormat.Unicode, rtol=1e-4) == "1/2"


def test_density_matrix_draw_rtol() -> None:
    # `rtol` is accepted by `DensityMatrix.draw` and threaded to the entry recognition.
    dm = DensityMatrix(data=[BasicStates.PLUS])
    assert dm.draw(OutputFormat.Unicode) == "[ 1/2  1/2 ]\n[ 1/2  1/2 ]"
    assert dm.draw(OutputFormat.Unicode, rtol=1e-4) == "[ 1/2  1/2 ]\n[ 1/2  1/2 ]"


def test_statevec_draw_factor_relative_phase() -> None:
    # A modulus shared up to a relative phase is still factored, with the phase kept
    # inside the parentheses.
    assert Statevector(data=BasicStates.PLUS).draw(OutputFormat.Unicode) == "√2/2(|0⟩ + |1⟩)"
    assert Statevector(data=BasicStates.PLUS_I).draw(OutputFormat.Unicode) == "√2/2(|0⟩ + i|1⟩)"
    assert Statevector(data=BasicStates.MINUS_I).draw(OutputFormat.Unicode) == "√2/2(|0⟩ - i|1⟩)"
    assert (
        Statevector(data=BasicStates.PLUS_I).draw(OutputFormat.LaTeX)
        == r"\frac{\sqrt{2}}{2}(\ket{0} + \mathrm{i}\ket{1})"
    )


def test_factor_uniform_magnitude_edge_cases() -> None:
    kets = ["|0⟩", "|1⟩"]
    # Non-numeric (e.g. symbolic) amplitudes have no modulus -> None (term-by-term fallback).
    assert (
        _factor_uniform_magnitude(
            ["alpha", "beta"], kets, OutputFormat.Unicode, max_denominator=1000, atol=1e-9, rtol=0.0, precision=4
        )
        is None
    )
    # Zero modulus -> None.
    assert (
        _factor_uniform_magnitude(
            [0, 0], kets, OutputFormat.Unicode, max_denominator=1000, atol=1e-9, rtol=0.0, precision=4
        )
        is None
    )
    # A unit modulus is not worth factoring -> None.
    assert (
        _factor_uniform_magnitude(
            [1, 1j], kets, OutputFormat.Unicode, max_denominator=1000, atol=1e-9, rtol=0.0, precision=4
        )
        is None
    )
    # A non-nice relative phase still factors, with the phase as a decimal coefficient.
    factored = _factor_uniform_magnitude(
        [0.5, 0.5 * cmath.exp(1j * 0.3)],
        kets,
        OutputFormat.Unicode,
        max_denominator=1000,
        atol=1e-9,
        rtol=0.0,
        precision=4,
    )
    assert factored is not None
    assert factored.startswith("1/2(|0⟩ + ")


def test_draw_max_denominator() -> None:
    # `max_denominator` caps the denominators the recognition will accept.
    dm = DensityMatrix(data=[BasicStates.PLUS])
    assert dm.draw(OutputFormat.Unicode) == "[ 1/2  1/2 ]\n[ 1/2  1/2 ]"
    # `max_denominator=2` is the smallest cap that still recognizes 1/2,
    # matching the user-facing semantics ("denominators up to N in the output").
    assert dm.draw(OutputFormat.Unicode, max_denominator=2) == "[ 1/2  1/2 ]\n[ 1/2  1/2 ]"
    # With max_denominator=1, 1/2 can no longer be recognized and falls back to a decimal.
    assert dm.draw(OutputFormat.Unicode, max_denominator=1) == "[ 0.5  0.5 ]\n[ 0.5  0.5 ]"
    # Same effect on the statevector draw: √2/2 needs max_denominator >= 2 for the
    # squared form 1/2 to be representable; below that it falls back to 0.7071.
    sv = Statevector([2**-0.5, 2**-0.5])
    assert sv.draw(OutputFormat.Unicode) == "√2/2(|0⟩ + |1⟩)"
    assert sv.draw(OutputFormat.Unicode, max_denominator=2) == "√2/2(|0⟩ + |1⟩)"
    assert sv.draw(OutputFormat.Unicode, max_denominator=1) == "0.7071(|0⟩ + |1⟩)"


def test_complex_to_str_edge_cases() -> None:
    # A tiny non-zero real collapses to "0" (zero branches of the square-free
    # decomposition and the real renderer).
    assert complex_to_str(1e-7, OutputFormat.Unicode) == "0"
    # A purely imaginary value with no nice form falls back to a decimal imaginary part.
    assert complex_to_str(0.234567j, OutputFormat.Unicode) == "0.2346i"
    # A recognized phase with an unrecognized radius (π·e^(iπ/4)) cannot use the exponential
    # or Cartesian forms and falls back to a decimal.
    assert complex_to_str(math.pi * cmath.exp(1j * math.pi / 4), OutputFormat.Unicode) == "2.221+2.221i"


def test_statevec_draw_all_amplitudes_dropped() -> None:
    # A tolerance large enough to drop every amplitude prints the statevector as "0".
    bell = Statevector([2**-0.5, 0, 0, 2**-0.5])
    assert bell.draw(OutputFormat.Unicode, atol=1.0) == "0"


def test_statevec_draw_non_uniform_not_factored() -> None:
    # Amplitudes with different magnitudes are not factored; each term keeps its coefficient.
    # The negative second amplitude also exercises the ` - ` separator in the term-by-term
    # fallback (the factoring path handles the uniform-magnitude negative case separately).
    state = Statevector([math.sqrt(3) / 2, -0.5])
    assert state.draw(OutputFormat.Unicode) == "√3/2|0⟩ - 1/2|1⟩"
