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
from graphix.pretty_print import OutputFormat, complex_to_str, densitymatrix_to_str, pattern_to_str, statevec_to_str
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


class TestComplexToStr:
    """Tests for :func:`~graphix.pretty_print.complex_to_str`."""

    def test_zero(self) -> None:
        assert complex_to_str(0j, OutputFormat.Unicode) == "0"
        assert complex_to_str(0j, OutputFormat.ASCII) == "0"
        assert complex_to_str(0j, OutputFormat.LaTeX) == "0"

    def test_integer(self) -> None:
        assert complex_to_str(1 + 0j, OutputFormat.Unicode) == "1"
        assert complex_to_str(-1 + 0j, OutputFormat.Unicode) == "-1"
        assert complex_to_str(2 + 0j, OutputFormat.Unicode) == "2"

    def test_simple_fractions(self) -> None:
        assert complex_to_str(0.5 + 0j, OutputFormat.Unicode) == "1/2"
        assert complex_to_str(0.25 + 0j, OutputFormat.Unicode) == "1/4"
        assert complex_to_str(0.75 + 0j, OutputFormat.Unicode) == "3/4"
        assert complex_to_str(-0.5 + 0j, OutputFormat.Unicode) == "-1/2"

    def test_square_roots(self) -> None:
        sqrt2_inv = 1 / math.sqrt(2)
        sqrt3_2 = math.sqrt(3) / 2
        assert complex_to_str(sqrt2_inv, OutputFormat.Unicode) == "1/√2", f"got {complex_to_str(sqrt2_inv, OutputFormat.Unicode)!r}"
        assert complex_to_str(sqrt3_2, OutputFormat.Unicode) == "√3/2"
        assert complex_to_str(-sqrt2_inv, OutputFormat.Unicode) == "-1/√2"

    def test_exponential(self) -> None:
        e_ipi_3 = 0.5 + 0.8660254037844386j
        e_ipi_4 = math.sqrt(2) / 2 + 1j * math.sqrt(2) / 2
        e_i2pi_3 = -0.5 + 0.8660254037844386j
        assert complex_to_str(e_ipi_3, OutputFormat.Unicode) == "e^(iπ/3)", f"got {complex_to_str(e_ipi_3, OutputFormat.Unicode)!r}"
        assert complex_to_str(e_ipi_4, OutputFormat.Unicode) == "e^(iπ/4)", f"got {complex_to_str(e_ipi_4, OutputFormat.Unicode)!r}"
        assert complex_to_str(e_i2pi_3, OutputFormat.Unicode) == "e^(i2π/3)", f"got {complex_to_str(e_i2pi_3, OutputFormat.Unicode)!r}"

    def test_pure_imaginary(self) -> None:
        assert complex_to_str(1j, OutputFormat.Unicode) == "i"
        assert complex_to_str(-1j, OutputFormat.Unicode) == "-i"
        assert complex_to_str(0.5j, OutputFormat.Unicode) == "1/2i"
        assert complex_to_str(-0.5j, OutputFormat.Unicode) == "-1/2i"

    def test_general_complex(self) -> None:
        assert complex_to_str(1 + 1j, OutputFormat.Unicode) == "1+i"
        assert complex_to_str(1 - 1j, OutputFormat.Unicode) == "1-i"
        assert complex_to_str(-1 + 1j, OutputFormat.Unicode) == "-1+i"
        assert complex_to_str(0.5 + 0.5j, OutputFormat.Unicode) == "1/2+1/2i"

    def test_ascii_format(self) -> None:
        assert complex_to_str(1j, OutputFormat.ASCII) == "i"
        assert complex_to_str(0.7071067811865474 + 0j, OutputFormat.ASCII) == "1/sqrt(2)"
        assert complex_to_str(0.5 + 0.8660254037844386j, OutputFormat.ASCII) == "exp(i*pi/3)"

    def test_latex_format(self) -> None:
        assert complex_to_str(1j, OutputFormat.LaTeX) == r"\mathrm{i}"
        assert complex_to_str(0.5 + 0j, OutputFormat.LaTeX) == r"\frac{1}{2}"
        assert complex_to_str(0.7071067811865474 + 0j, OutputFormat.LaTeX) == r"\frac{1}{\sqrt{2}}"
        assert complex_to_str(0.5 + 0.8660254037844386j, OutputFormat.LaTeX) == r"\mathrm{e}^{\mathrm{i}\pi/3}"


class TestStatevecToStr:
    """Tests for :func:`~graphix.pretty_print.statevec_to_str`."""

    def test_single_basis_state(self) -> None:
        sv = Statevec(data=[BasicStates.ZERO])
        d = sv.to_dict()
        assert statevec_to_str(d, OutputFormat.Unicode) == "|0⟩"

    def test_two_qubit_product(self) -> None:
        sv = Statevec(data=[BasicStates.ZERO, BasicStates.ONE])
        d = sv.to_dict()
        result = statevec_to_str(d, OutputFormat.Unicode)
        assert result == "|01⟩"

    def test_plus_state(self) -> None:
        sv = Statevec(data=[BasicStates.PLUS])
        d = sv.to_dict()
        result = statevec_to_str(d, OutputFormat.Unicode)
        assert result == "1/√2|0⟩ + 1/√2|1⟩"

    def test_latex_statevec(self) -> None:
        sv = Statevec(data=[BasicStates.PLUS])
        d = sv.to_dict()
        result = statevec_to_str(d, OutputFormat.LaTeX)
        assert result == r"\frac{1}{\sqrt{2}}\lvert 0\rangle + \frac{1}{\sqrt{2}}\lvert 1\rangle"

    def test_empty_dict(self) -> None:
        assert statevec_to_str({}, OutputFormat.Unicode) == "0"


class TestDensityMatrixToStr:
    """Tests for :func:`~graphix.pretty_print.densitymatrix_to_str`."""

    def test_pure_state_zero(self) -> None:
        dm = DensityMatrix(data=[BasicStates.ZERO])
        result = densitymatrix_to_str(dm.rho, dm.nqubit, OutputFormat.Unicode)
        assert result == "|0⟩⟨0|"

    def test_mixed_state(self) -> None:
        """Check a 50/50 mixed state returns a sum of projectors."""
        rho = np.eye(2, dtype=np.complex128) / 2
        result = densitymatrix_to_str(rho, 1, OutputFormat.Unicode)
        assert "1/2" in result
        assert result.count("⟩⟨") == 2


class TestStatevecDraw:
    """Tests for :meth:`graphix.sim.statevec.Statevec.draw`."""

    def test_draw(self, capsys: pytest.CaptureFixture[str]) -> None:
        sv = Statevec(data=[BasicStates.PLUS])
        sv.draw()
        captured = capsys.readouterr()
        assert "1/√2|0⟩ + 1/√2|1⟩" in captured.out


class TestDensityMatrixDraw:
    """Tests for :meth:`graphix.sim.density_matrix.DensityMatrix.draw`."""

    def test_draw(self, capsys: pytest.CaptureFixture[str]) -> None:
        dm = DensityMatrix(data=[BasicStates.ZERO])
        dm.draw()
        captured = capsys.readouterr()
        assert "|0⟩⟨0|" in captured.out
