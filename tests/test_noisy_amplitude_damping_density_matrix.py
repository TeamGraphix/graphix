from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest

from graphix.branch_selector import ConstBranchSelector, FixedBranchSelector
from graphix.channels import amplitude_damping_channel
from graphix.command import CommandKind
from graphix.fundamentals import angle_to_rad
from graphix.measurements import Measurement
from graphix.noise_models import AmplitudeDampingNoise, AmplitudeDampingNoiseModel, TwoQubitAmplitudeDampingNoise
from graphix.noise_models.noise_model import NoiselessNoiseModel
from graphix.ops import Ops
from graphix.sim.base_backend import _outcome_to_operator_matrix
from graphix.sim.density_matrix import DensityMatrix
from graphix.states import BasicStates
from graphix.transpiler import Circuit

if TYPE_CHECKING:
    from numpy.random import Generator

    from graphix.fundamentals import Angle
    from graphix.measurements import Outcome
    from graphix.noise_models.noise_model import Noise
    from graphix.pattern import Pattern


def apply_amplitude_damping(rho: npt.NDArray[np.complex128], gamma: float) -> npt.NDArray[np.complex128]:
    """Apply single-qubit amplitude damping to a density matrix."""
    dm = DensityMatrix(data=rho)
    dm.apply_channel(amplitude_damping_channel(gamma), [0])
    return dm.rho.astype(np.complex128)


class _ReferenceDensityMatrixSimulator:
    """Minimal density-matrix simulator for independent test references."""

    def __init__(self) -> None:
        self.dm = DensityMatrix(nqubit=0)
        self.node_index: list[int] = []

    def add_nodes(self, nodes: list[int]) -> None:
        self.dm.add_nodes(len(nodes), BasicStates.PLUS)
        self.node_index.extend(nodes)

    def _loc(self, node: int) -> int:
        return self.node_index.index(node)

    def apply_noise(self, nodes: list[int], noise: Noise) -> None:
        self.dm.apply_noise([self._loc(node) for node in nodes], noise)

    def entangle(self, edge: tuple[int, int]) -> None:
        self.dm.entangle((self._loc(edge[0]), self._loc(edge[1])))

    def measure(self, node: int, measurement: Measurement, outcome: Outcome) -> None:
        bloch = measurement.to_bloch()
        vec = bloch.plane.polar(bloch.angle)
        operator = _outcome_to_operator_matrix(vec, outcome, symbolic=False)
        loc = self._loc(node)
        self.dm.evolve_single(operator, loc)
        self.dm.remove_qubit(loc)
        self.node_index.remove(node)

    def correct(self, node: int, operator: npt.NDArray[np.complex128]) -> None:
        self.dm.evolve_single(operator, self._loc(node))


def reference_hadamard_rho(
    *,
    prepare_gamma: float = 0.0,
    entangle_gamma: float = 0.0,
    measure_gamma: float = 0.0,
    x_gamma: float = 0.0,
    outcome: Outcome = 0,
) -> npt.NDArray[np.complex128]:
    """Return the expected output density matrix for the Hadamard pattern."""
    sim = _ReferenceDensityMatrixSimulator()
    sim.add_nodes([0])
    if prepare_gamma:
        sim.apply_noise([0], AmplitudeDampingNoise(prepare_gamma))
    sim.add_nodes([1])
    if prepare_gamma:
        sim.apply_noise([1], AmplitudeDampingNoise(prepare_gamma))
    sim.entangle((0, 1))
    if entangle_gamma:
        sim.apply_noise([0, 1], TwoQubitAmplitudeDampingNoise(entangle_gamma))
    if measure_gamma:
        sim.apply_noise([0], AmplitudeDampingNoise(measure_gamma))
    sim.measure(0, Measurement.X, outcome)
    if outcome == 1:
        sim.correct(1, Ops.X)
        if x_gamma:
            sim.apply_noise([1], AmplitudeDampingNoise(x_gamma))
    return sim.dm.rho.astype(np.complex128)


def reference_rz_rho(
    alpha: Angle,
    *,
    prepare_gamma: float = 0.0,
    entangle_gamma: float = 0.0,
    measure_gamma: float = 0.0,
    x_gamma: float = 0.0,
    z_gamma: float = 0.0,
    m0_outcome: Outcome = 0,
    m1_outcome: Outcome = 0,
) -> npt.NDArray[np.complex128]:
    """Return the expected output density matrix for the RZ pattern."""
    sim = _ReferenceDensityMatrixSimulator()
    sim.add_nodes([0])
    if prepare_gamma:
        sim.apply_noise([0], AmplitudeDampingNoise(prepare_gamma))
    sim.add_nodes([1])
    if prepare_gamma:
        sim.apply_noise([1], AmplitudeDampingNoise(prepare_gamma))
    sim.add_nodes([2])
    if prepare_gamma:
        sim.apply_noise([2], AmplitudeDampingNoise(prepare_gamma))
    sim.entangle((0, 1))
    if entangle_gamma:
        sim.apply_noise([0, 1], TwoQubitAmplitudeDampingNoise(entangle_gamma))
    sim.entangle((1, 2))
    if entangle_gamma:
        sim.apply_noise([1, 2], TwoQubitAmplitudeDampingNoise(entangle_gamma))
    if measure_gamma:
        sim.apply_noise([0], AmplitudeDampingNoise(measure_gamma))
    sim.measure(0, Measurement.XY(-alpha), m0_outcome)
    if measure_gamma:
        sim.apply_noise([1], AmplitudeDampingNoise(measure_gamma))
    sim.measure(1, Measurement.X, m1_outcome)
    if m1_outcome == 1:
        sim.correct(2, Ops.X)
        if x_gamma:
            sim.apply_noise([2], AmplitudeDampingNoise(x_gamma))
    if m0_outcome == 1:
        sim.correct(2, Ops.Z)
        if z_gamma:
            sim.apply_noise([2], AmplitudeDampingNoise(z_gamma))
    return sim.dm.rho.astype(np.complex128)


def rz_exact_res(alpha: Angle) -> npt.NDArray[np.float64]:
    rad = angle_to_rad(alpha)
    return 0.5 * np.array([[1, np.exp(-1j * rad)], [np.exp(1j * rad), 1]])


def hpat() -> Pattern:
    circ = Circuit(1)
    circ.h(0)
    return circ.transpile().pattern


def rzpat(alpha: Angle) -> Pattern:
    circ = Circuit(1)
    circ.rz(0, alpha)
    return circ.transpile().pattern


class TestNoisyAmplitudeDampingDensityMatrixBackend:
    """Test amplitude damping in the density-matrix backend."""

    @pytest.mark.filterwarnings("ignore:Simulating using densitymatrix backend with no noise.")
    def test_noiseless_noisy_hadamard(self, fx_rng: Generator) -> None:
        hadamardpattern = hpat()
        noiselessres = hadamardpattern.simulate_pattern(backend="densitymatrix", rng=fx_rng)
        noisynoiselessres = hadamardpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=NoiselessNoiseModel(),
            rng=fx_rng,
        )
        assert isinstance(noiselessres, DensityMatrix)
        assert np.allclose(noiselessres.rho, np.array([[1.0, 0.0], [0.0, 0.0]]))
        assert isinstance(noisynoiselessres, DensityMatrix)
        assert np.allclose(noisynoiselessres.rho, np.array([[1.0, 0.0], [0.0, 0.0]]))

    def test_noisy_measure_confuse_hadamard(self, fx_rng: Generator) -> None:
        hadamardpattern = hpat()
        res = hadamardpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=AmplitudeDampingNoiseModel(measure_error_prob=1.0),
            rng=fx_rng,
        )
        assert isinstance(res, DensityMatrix)
        assert np.allclose(res.rho, np.array([[0.0, 0.0], [0.0, 1.0]]))

    @pytest.mark.parametrize("outcome", [0, 1])
    def test_noisy_measure_confuse_hadamard_arbitrary(self, fx_rng: Generator, outcome: Outcome) -> None:
        hadamardpattern = hpat()
        measure_error_pr = fx_rng.random()
        res = hadamardpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=AmplitudeDampingNoiseModel(measure_error_prob=measure_error_pr),
            branch_selector=ConstBranchSelector(outcome),
            rng=fx_rng,
        )
        assert isinstance(res, DensityMatrix)
        assert np.allclose(res.rho, np.array([[1.0, 0.0], [0.0, 0.0]])) or np.allclose(
            res.rho,
            np.array([[0.0, 0.0], [0.0, 1.0]]),
        )

    def test_noisy_measure_channel_hadamard(self, fx_rng: Generator) -> None:
        hadamardpattern = hpat()
        measure_channel_gamma = fx_rng.random()
        res = hadamardpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=AmplitudeDampingNoiseModel(measure_channel_prob=measure_channel_gamma),
            rng=fx_rng,
        )
        expected = reference_hadamard_rho(measure_gamma=measure_channel_gamma)
        assert isinstance(res, DensityMatrix)
        assert np.allclose(res.rho, expected)

    @pytest.mark.parametrize("outcome", [0, 1])
    def test_noisy_x_hadamard(self, fx_rng: Generator, outcome: Outcome) -> None:
        hadamardpattern = hpat()
        x_error_gamma = fx_rng.random()
        res = hadamardpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=AmplitudeDampingNoiseModel(x_error_prob=x_error_gamma),
            branch_selector=ConstBranchSelector(outcome),
            rng=fx_rng,
        )
        expected = reference_hadamard_rho(x_gamma=x_error_gamma, outcome=outcome)
        assert isinstance(res, DensityMatrix)
        assert np.allclose(res.rho, expected)

    @pytest.mark.parametrize("outcome", [0, 1])
    def test_noisy_entanglement_hadamard(self, fx_rng: Generator, outcome: Outcome) -> None:
        hadamardpattern = hpat()
        entanglement_error_gamma = fx_rng.uniform()
        res = hadamardpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=AmplitudeDampingNoiseModel(entanglement_error_prob=entanglement_error_gamma),
            branch_selector=ConstBranchSelector(outcome),
            rng=fx_rng,
        )
        expected = reference_hadamard_rho(entangle_gamma=entanglement_error_gamma, outcome=outcome)
        assert isinstance(res, DensityMatrix)
        assert np.allclose(res.rho, expected)

    @pytest.mark.parametrize("outcome", [0, 1])
    def test_noisy_preparation_hadamard(self, fx_rng: Generator, outcome: Outcome) -> None:
        hadamardpattern = hpat()
        prepare_error_gamma = fx_rng.random()
        res = hadamardpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=AmplitudeDampingNoiseModel(prepare_error_prob=prepare_error_gamma),
            branch_selector=ConstBranchSelector(outcome),
            rng=fx_rng,
        )
        expected = reference_hadamard_rho(prepare_gamma=prepare_error_gamma, outcome=outcome)
        assert isinstance(res, DensityMatrix)
        assert np.allclose(res.rho, expected)

    def test_amplitude_damping_channel_preserves_zero(self, fx_rng: Generator) -> None:
        gamma = fx_rng.random()
        rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        assert np.allclose(apply_amplitude_damping(rho, gamma), rho)

    def test_amplitude_damping_channel_on_one_state(self, fx_rng: Generator) -> None:
        gamma = fx_rng.random()
        rho = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
        expected = np.array([[gamma, 0.0], [0.0, 1.0 - gamma]])
        assert np.allclose(apply_amplitude_damping(rho, gamma), expected)

    @pytest.mark.filterwarnings("ignore:Simulating using densitymatrix backend with no noise.")
    def test_noiseless_noisy_rz(self, fx_rng: Generator) -> None:
        alpha = fx_rng.random()
        rzpattern = rzpat(alpha)
        noiselessres = rzpattern.simulate_pattern(backend="densitymatrix", rng=fx_rng)
        noisynoiselessres = rzpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=AmplitudeDampingNoiseModel(),
            rng=fx_rng,
        )
        assert isinstance(noiselessres, DensityMatrix)
        assert np.allclose(noiselessres.rho, rz_exact_res(alpha))
        assert isinstance(noisynoiselessres, DensityMatrix)
        assert np.allclose(noisynoiselessres.rho, rz_exact_res(alpha))

    @pytest.mark.parametrize("outcome", [0, 1])
    def test_noisy_preparation_rz(self, fx_rng: Generator, outcome: Outcome) -> None:
        alpha = fx_rng.random()
        rzpattern = rzpat(alpha)
        prepare_error_gamma = fx_rng.random()
        res = rzpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=AmplitudeDampingNoiseModel(prepare_error_prob=prepare_error_gamma),
            branch_selector=ConstBranchSelector(outcome),
            rng=fx_rng,
        )
        expected = reference_rz_rho(alpha, prepare_gamma=prepare_error_gamma, m0_outcome=outcome, m1_outcome=outcome)
        assert isinstance(res, DensityMatrix)
        assert np.allclose(res.rho, expected)

    @pytest.mark.parametrize("outcome", [0, 1])
    def test_noisy_entanglement_rz(self, fx_rng: Generator, outcome: Outcome) -> None:
        alpha = fx_rng.random()
        rzpattern = rzpat(alpha)
        entanglement_error_gamma = fx_rng.uniform()
        res = rzpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=AmplitudeDampingNoiseModel(entanglement_error_prob=entanglement_error_gamma),
            branch_selector=ConstBranchSelector(outcome),
            rng=fx_rng,
        )
        expected = reference_rz_rho(
            alpha, entangle_gamma=entanglement_error_gamma, m0_outcome=outcome, m1_outcome=outcome
        )
        assert isinstance(res, DensityMatrix)
        assert np.allclose(res.rho, expected)

    def test_noisy_measure_channel_rz(self, fx_rng: Generator) -> None:
        alpha = fx_rng.random()
        rzpattern = rzpat(alpha)
        measure_channel_gamma = fx_rng.random()
        res = rzpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=AmplitudeDampingNoiseModel(measure_channel_prob=measure_channel_gamma),
            rng=fx_rng,
        )
        expected = reference_rz_rho(alpha, measure_gamma=measure_channel_gamma)
        assert isinstance(res, DensityMatrix)
        assert np.allclose(res.rho, expected)

    @pytest.mark.parametrize("x_outcome", [0, 1])
    def test_noisy_x_rz(self, fx_rng: Generator, x_outcome: Outcome) -> None:
        alpha = fx_rng.random()
        rzpattern = rzpat(alpha)
        x_error_gamma = fx_rng.random()
        m_nodes = (cmd.node for cmd in rzpattern if cmd.kind == CommandKind.M)
        results: dict[int, Outcome] = {next(m_nodes): 0, next(m_nodes): x_outcome}
        res = rzpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=AmplitudeDampingNoiseModel(x_error_prob=x_error_gamma),
            branch_selector=FixedBranchSelector(results),
            rng=fx_rng,
        )
        expected = reference_rz_rho(alpha, x_gamma=x_error_gamma, m1_outcome=x_outcome)
        assert isinstance(res, DensityMatrix)
        assert np.allclose(res.rho, expected)

    @pytest.mark.parametrize("outcome_z", [0, 1])
    @pytest.mark.parametrize("outcome_x", [0, 1])
    def test_noisy_z_rz(self, fx_rng: Generator, outcome_z: Outcome, outcome_x: Outcome) -> None:
        alpha = fx_rng.random()
        rzpattern = rzpat(alpha)
        z_error_gamma = fx_rng.random()
        results: dict[int, Outcome] = {0: outcome_z, 1: outcome_x}
        res = rzpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=AmplitudeDampingNoiseModel(z_error_prob=z_error_gamma),
            branch_selector=FixedBranchSelector(results),
            rng=fx_rng,
        )
        expected = reference_rz_rho(alpha, z_gamma=z_error_gamma, m0_outcome=outcome_z, m1_outcome=outcome_x)
        assert isinstance(res, DensityMatrix)
        assert np.allclose(res.rho, expected)

    @pytest.mark.parametrize("z_outcome", [0, 1])
    @pytest.mark.parametrize("x_outcome", [0, 1])
    def test_noisy_measure_confuse_rz(self, fx_rng: Generator, z_outcome: Outcome, x_outcome: Outcome) -> None:
        alpha = fx_rng.random()
        rzpattern = rzpat(alpha)
        results: dict[int, Outcome] = {0: z_outcome, 1: x_outcome}
        res = rzpattern.simulate_pattern(
            backend="densitymatrix",
            noise_model=AmplitudeDampingNoiseModel(measure_error_prob=1.0),
            branch_selector=FixedBranchSelector(results),
            rng=fx_rng,
        )
        exact = rz_exact_res(alpha)
        assert isinstance(res, DensityMatrix)
        assert np.allclose(res.rho, Ops.Z @ Ops.X @ exact @ Ops.X @ Ops.Z)
