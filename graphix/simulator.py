"""MBQC simulator.

Simulates MBQC by executing the pattern.

"""

from __future__ import annotations

import abc
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

# assert_never introduced in Python 3.11
# override introduced in Python 3.12
from typing_extensions import assert_never, override

from graphix import command
from graphix.branch_selector import BranchSelector, RandomBranchSelector
from graphix.clifford import Clifford
from graphix.command import BaseM, CommandKind, MeasureUpdate, N
from graphix.measurements import Measurement, Outcome
from graphix.sim.base_backend import Backend
from graphix.sim.density_matrix import DensityMatrixBackend
from graphix.sim.statevec import StatevectorBackend
from graphix.sim.tensornet import TensorNetworkBackend
from graphix.states import BasicStates

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from numpy.random import Generator

    from graphix.command import BaseN
    from graphix.noise_models.noise_model import CommandOrNoise, NoiseModel
    from graphix.pattern import Pattern
    from graphix.sim import Backend, Data, DensityMatrix, DensityMatrixBackend, Statevec, StatevectorBackend
    from graphix.sim.base_backend import _StateT_co
    from graphix.sim.tensornet import MBQCTensorNet, TensorNetworkBackend


class PrepareMethod(abc.ABC):
    """Prepare method used by the simulator.

    See `DefaultPrepareMethod` for the default prepare method that implements MBQC.

    To be overwritten by custom preparation methods in the case of delegated QC protocols.

    Example: class `ClientPrepareMethod` in https://github.com/qat-inria/veriphix
    """

    @abc.abstractmethod
    def prepare(self, backend: Backend[_StateT_co], cmd: BaseN, rng: Generator | None = None) -> None:
        """Prepare a node."""


class DefaultPrepareMethod(PrepareMethod):
    """Default prepare method implementing standard preparation for MBQC."""

    @override
    def prepare(self, backend: Backend[_StateT_co], cmd: BaseN, rng: Generator | None = None) -> None:
        """Prepare a node."""
        if not isinstance(cmd, N):
            raise TypeError("The default prepare method requires all preparation commands to be of type `N`.")
        backend.add_nodes(nodes=[cmd.node], data=cmd.state)


class MeasureMethod(abc.ABC):
    """Measure method used by the simulator, with default measurement method that implements MBQC.

    To be overwritten by custom measurement methods in the case of delegated QC protocols.

    Example: class `ClientMeasureMethod` in https://github.com/qat-inria/veriphix
    """

    def measure(
        self,
        backend: Backend[_StateT_co],
        cmd: BaseM,
        noise_model: NoiseModel | None = None,
        rng: Generator | None = None,
    ) -> None:
        """Perform a measure."""
        description = self.get_measurement_description(cmd)
        result = backend.measure(cmd.node, description, rng=rng)
        if noise_model is not None:
            result = noise_model.confuse_result(cmd, result, rng=rng)
        self.set_measure_result(cmd.node, result)

    @abc.abstractmethod
    def get_measurement_description(self, cmd: BaseM) -> Measurement:
        """Return the description of the measurement performed by a command.

        Parameters
        ----------
        cmd : BaseM
            Measurement command whose description is required.

        Returns
        -------
        Measurement
            Plane and angle actually used by the backend.
        """
        ...

    @abc.abstractmethod
    def get_measure_result(self, node: int) -> Outcome:
        """Return the result of a previous measurement.

        Parameters
        ----------
        node : int
            Node label of the measured qubit.

        Returns
        -------
        bool
            Recorded measurement outcome.
        """
        ...

    @abc.abstractmethod
    def set_measure_result(self, node: int, result: Outcome) -> None:
        """Store the result of a previous measurement.

        Parameters
        ----------
        node : int
            Node label of the measured qubit.
        result : bool
            Measurement outcome to store.
        """
        ...


class DefaultMeasureMethod(MeasureMethod):
    """Default measurement method implementing standard measurement plane/angle update for MBQC."""

    results: dict[int, Outcome]

    def __init__(self, results: Mapping[int, Outcome] | None = None):
        """Initialize with an optional result dictionary.

        Parameters
        ----------
        results : Mapping[int, Outcome] | None, optional
            Mapping of previously measured nodes to their results. If ``None``,
            an empty dictionary is created.

        Notes
        -----
        If a mapping is provided, it is treated as read-only. Measurements
        performed during simulation are stored in `self.results`, which is a copy
        of the given mapping. The original `results` mapping is not modified.
        """
        # results is coerced into dict, since `set_measure_result` mutates it.
        self.results = {} if results is None else dict(results)

    def get_measurement_description(self, cmd: BaseM) -> Measurement:
        """Return the description of the measurement performed by ``cmd``.

        Parameters
        ----------
        cmd : BaseM
            Measurement command whose plane and angle should be updated.

        Returns
        -------
        Measurement
            Updated measurement specification.
        """
        assert isinstance(cmd, command.M)
        angle = cmd.angle * np.pi
        # extract signals for adaptive angle
        s_signal = sum(self.results[j] for j in cmd.s_domain)
        t_signal = sum(self.results[j] for j in cmd.t_domain)
        measure_update = MeasureUpdate.compute(cmd.plane, s_signal % 2 == 1, t_signal % 2 == 1, Clifford.I)
        angle = angle * measure_update.coeff + measure_update.add_term
        return Measurement(angle, measure_update.new_plane)

    def get_measure_result(self, node: int) -> Outcome:
        """Return the result of a previous measurement.

        Parameters
        ----------
        node : int
            Node label of the measured qubit.

        Returns
        -------
        Outcome
            Stored measurement outcome.
        """
        return self.results[node]

    def set_measure_result(self, node: int, result: Outcome) -> None:
        """Store the result of a previous measurement.

        Parameters
        ----------
        node : int
            Node label of the measured qubit.
        result : bool
            Measurement outcome to store.
        """
        self.results[node] = result


class PatternSimulator:
    """MBQC simulator.

    Executes the measurement pattern.
    """

    noise_model: NoiseModel | None

    def __init__(
        self,
        pattern: Pattern,
        backend: Backend[_StateT_co] | str = "statevector",
        prepare_method: PrepareMethod | None = None,
        measure_method: MeasureMethod | None = None,
        noise_model: NoiseModel | None = None,
        branch_selector: BranchSelector | None = None,
        graph_prep: str | None = None,
        symbolic: bool = False,
    ) -> None:
        """
        Construct a pattern simulator.

        Parameters
        ----------
        pattern: :class:`Pattern` object
            MBQC pattern to be simulated.
        backend: :class:`Backend` object,
            or 'statevector', or 'densitymatrix', or 'tensornetwork'
            simulation backend (optional), default is 'statevector'.
        prepare_method: :class:`PrepareMethod`, optional
            Prepare method used by the simulator. Default is :class:`DefaultPrepareMethod`.
        measure_method: :class:`MeasureMethod`, optional
            Measure method used by the simulator. Default is :class:`DefaultMeasureMethod`.
        noise_model: :class:`NoiseModel`, optional
            [Density matrix backend only] Noise model used by the simulator.
        branch_selector: :class:`BranchSelector`, optional
            Branch selector used for measurements. Can only be specified if ``backend`` is not an already instantiated :class:`Backend` object.  Default is :class:`RandomBranchSelector`.
        graph_prep: str, optional
            [Tensor network backend only] Strategy for preparing the graph state.  See :class:`TensorNetworkBackend`.
        symbolic : bool, optional
            [State vector and density matrix backends only] If True, support arbitrary objects (typically, symbolic expressions) in measurement angles.

        .. seealso:: :class:`graphix.sim.statevec.StatevectorBackend`\
            :class:`graphix.sim.tensornet.TensorNetworkBackend`\
            :class:`graphix.sim.density_matrix.DensityMatrixBackend`\
        """
        self.backend = self.initialize_backend(pattern, backend, noise_model, branch_selector, graph_prep, symbolic)
        self.noise_model = noise_model
        self.__pattern = pattern
        if prepare_method is None:
            prepare_method = DefaultPrepareMethod()
        self.__prepare_method = prepare_method
        if measure_method is None:
            measure_method = DefaultMeasureMethod(pattern.results)
        self.__measure_method = measure_method

    @property
    def pattern(self) -> Pattern:
        """Return the pattern."""
        return self.__pattern

    @property
    def measure_method(self) -> MeasureMethod:
        """Return the measure method."""
        return self.__measure_method

    @staticmethod
    def initialize_backend(pattern: Pattern, backend: Backend[_StateT_co] | str, noise_model: NoiseModel | None, branch_selector: BranchSelector | None, graph_prep: str | None, symbolic: bool) -> Backend[Any]:
        """
        Initialize the backend.

        Parameters
        ----------
        backend: :class:`Backend` object,
            'statevector', or 'densitymatrix', or 'tensornetwork'
            simulation backend (optional), default is 'statevector'.
        noise_model: :class:`NoiseModel`, optional
            [Density matrix backend only] Noise model used by the simulator.
        branch_selector: :class:`BranchSelector`, optional
            Branch selector used for measurements. Can only be specified if ``backend`` is not an already instantiated :class:`Backend` object.  Default is :class:`RandomBranchSelector`.
        graph_prep: str, optional
            [Tensor network backend only] Strategy for preparing the graph state.  See :class:`TensorNetworkBackend`.
        symbolic : bool, optional
            [State vector and density matrix backends only] If True, support arbitrary objects (typically, symbolic expressions) in measurement angles.

        Returns
        -------
        :class:`Backend`
            matching the appropriate backend
        """
        if isinstance(backend, Backend):
            if branch_selector is not None:
                raise ValueError("`branch_selector` cannot be specified if `backend` is already instantiated.")
            if graph_prep is not None:
                raise ValueError("`graph_prep` cannot be specified if `backend` is already instantiated.")
            if symbolic:
                raise ValueError("`symbolic` cannot be specified if `backend` is already instantiated.")
            return backend
        if branch_selector is None:
            branch_selector = RandomBranchSelector()
        if backend in {"tensornetwork", "mps"}:
            if noise_model is not None:
                raise ValueError("`noise_model` cannot be specified for tensor network backend.")
            if symbolic:
                raise ValueError("`symbolic` cannot be specified for tensor network backend.")
            if graph_prep is None:
                graph_prep = "auto"
            return TensorNetworkBackend(pattern, branch_selector=branch_selector, graph_prep=graph_prep)
        if graph_prep is not None:
            raise ValueError("`graph_prep` can only be specified for tensor network backend.")
        if backend == "statevector":
            if noise_model is not None:
                raise ValueError("`noise_model` cannot be specified for state vector backend.")
            return StatevectorBackend(branch_selector=branch_selector, symbolic=symbolic)
        if backend == "densitymatrix":
            if noise_model is None:
                warnings.warn(
                    "Simulating using densitymatrix backend with no noise. To add noise to the simulation, give an object of `graphix.noise_models.Noisemodel` to `noise_model` keyword argument.",
                    stacklevel=1,
                )
            return DensityMatrixBackend(branch_selector=branch_selector, symbolic=symbolic)
        raise ValueError(f"Unknown backend {backend}.")

    def set_noise_model(self, model: NoiseModel | None) -> None:
        """Set a noise model."""
        self.noise_model = model

    def run(self, input_state: Data = BasicStates.PLUS, rng: Generator | None = None) -> None:
        """Perform the simulation.

        Returns
        -------
        input_state: Data, optional
            the output quantum state,
            in the representation depending on the backend used.
            Default: ``|+>``.
        rng: Generator, optional
            Random-number generator for measurements.
            This generator is used only in case of random branch selection
            (see :class:`RandomBranchSelector`).
        """
        if input_state is not None:
            self.backend.add_nodes(self.pattern.input_nodes, input_state)
        if self.noise_model is None:
            pattern: Iterable[CommandOrNoise] = self.pattern
        else:
            pattern = self.noise_model.input_nodes(self.pattern.input_nodes, rng=rng) if input_state is not None else []
            pattern.extend(self.noise_model.transpile(self.pattern, rng=rng))

        # We check runnability first to provide clearer error messages and
        # to catch these errors before starting the simulation.
        self.pattern.check_runnability()

        for cmd in pattern:
            if cmd.kind == CommandKind.N:
                self.__prepare_method.prepare(self.backend, cmd, rng=rng)
            elif cmd.kind == CommandKind.E:
                self.backend.entangle_nodes(edge=cmd.nodes)
            elif cmd.kind == CommandKind.M:
                self.__measure_method.measure(self.backend, cmd, noise_model=self.noise_model, rng=rng)
            # Use of `==` here for mypy
            elif cmd.kind == CommandKind.X or cmd.kind == CommandKind.Z:  # noqa: PLR1714
                self.backend.correct_byproduct(cmd, self.__measure_method)
            elif cmd.kind == CommandKind.C:
                self.backend.apply_clifford(cmd.node, cmd.clifford)
            elif cmd.kind == CommandKind.T:
                # The T command is a flag for one clock cycle in a simulated
                # experiment, added via a hardware-agnostic
                # pattern modifier. Noise models can perform special
                # handling of ticks during noise transpilation.
                pass
            elif cmd.kind == CommandKind.ApplyNoise:
                self.backend.apply_noise(cmd.nodes, cmd.noise)
            elif cmd.kind == CommandKind.S:
                raise ValueError("S commands unexpected in simulated patterns.")
            else:
                assert_never(cmd.kind)
        self.backend.finalize(output_nodes=self.pattern.output_nodes)
