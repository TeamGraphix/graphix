"""MBQC simulator.

Simulates MBQC by executing the pattern.

"""

from __future__ import annotations

import abc
import warnings
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from graphix import command
from graphix.branch_selector import BranchSelector, RandomBranchSelector
from graphix.clifford import Clifford
from graphix.command import BaseM, CommandKind, MeasureUpdate
from graphix.measurements import Measurement, Outcome
from graphix.sim.base_backend import Backend
from graphix.sim.density_matrix import DensityMatrixBackend
from graphix.sim.statevec import StatevectorBackend
from graphix.sim.tensornet import TensorNetworkBackend
from graphix.states import BasicStates

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.random import Generator

    from graphix.noise_models.noise_model import NoiseModel
    from graphix.pattern import Pattern
    from graphix.sim import BackendState, Data


_StateT_co = TypeVar("_StateT_co", bound="BackendState", covariant=True)


class MeasureMethod(abc.ABC):
    """Measure method used by the simulator, with default measurement method that implements MBQC.

    To be overwritten by custom measurement methods in the case of delegated QC protocols.

    Example: class `ClientMeasureMethod` in https://github.com/qat-inria/veriphix
    """

    def measure(self, backend: Backend[_StateT_co], cmd: BaseM, noise_model: NoiseModel | None = None) -> None:
        """Perform a measure."""
        description = self.get_measurement_description(cmd)
        result = backend.measure(cmd.node, description)
        if noise_model is not None:
            result = noise_model.confuse_result(result)
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
        backend: Backend[BackendState] | str = "statevector",
        measure_method: MeasureMethod | None = None,
        noise_model: NoiseModel | None = None,
        branch_selector: BranchSelector | None = None,
        rng: Generator | None = None,
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
        measure_method: :class:`MeasureMethod`, optional
            Measure method used by the simulator. Default is :class:`DefaultMeasureMethod`.
        noise_model: :class:`NoiseModel`, optional
            [Density matrix backend only] Noise model used by the simulator.
        branch_selector: :class:`BranchSelector`, optional
            Branch selector used for measurements. Can only be specified if ``backend`` is not an already instantiated :class:`Backend` object.  Default is :class:`RandomBranchSelector`.
        rng: :class:`numpy.random.Generator`, optional
            Random number generator to be used by the default :class:`RandomBranchSelector`. Can only be specified if ``backend`` is not an already instantiated :class:`Backend` object and if ``branch_selector`` is not specified.
        graph_prep: str, optional
            [Tensor network backend only] Strategy for preparing the graph state.  See :class:`TensorNetworkBackend`.
        symbolic : bool, optional
            [State vector and density matrix backends only] If True, support arbitrary objects (typically, symbolic expressions) in measurement angles.

        .. seealso:: :class:`graphix.sim.statevec.StatevectorBackend`\
            :class:`graphix.sim.tensornet.TensorNetworkBackend`\
            :class:`graphix.sim.density_matrix.DensityMatrixBackend`\
        """

        def initialize_backend() -> Backend[BackendState]:
            nonlocal backend, branch_selector, rng, graph_prep, noise_model
            if isinstance(backend, Backend):
                if noise_model is not None:
                    raise ValueError("`noise_model` cannot be specified if `backend` is already instantiated.")
                if branch_selector is not None:
                    raise ValueError("`branch_selector` cannot be specified if `backend` is already instantiated.")
                if rng is not None:
                    raise ValueError("`rng` cannot be specified if `backend` is already instantiated.")
                if graph_prep is not None:
                    raise ValueError("`graph_prep` cannot be specified if `backend` is already instantiated.")
                if symbolic:
                    raise ValueError("`symbolic` cannot be specified if `backend` is already instantiated.")
                return backend
            if branch_selector is None:
                branch_selector = RandomBranchSelector(rng=rng)
            elif rng is not None:
                raise ValueError("`rng` and `branch_selector` cannot be specified simultaneously.")
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

        self.backend = initialize_backend()
        self.noise_model = noise_model
        self.__pattern = pattern
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

    def set_noise_model(self, model: NoiseModel | None) -> None:
        """Set a noise model."""
        if not isinstance(self.backend, DensityMatrixBackend) and model is not None:
            self.noise_model = None  # if not initialized yet
            raise ValueError(f"The backend {self.backend} doesn't support noise but noisemodel was provided.")
        self.noise_model = model

    def run(self, input_state: Data = BasicStates.PLUS) -> None:
        """Perform the simulation.

        Returns
        -------
        state :
            the output quantum state,
            in the representation depending on the backend used.
        """
        if input_state is not None:
            self.backend.add_nodes(self.pattern.input_nodes, input_state)
        if self.noise_model is None:
            for cmd in self.pattern:
                if cmd.kind == CommandKind.N:
                    self.backend.add_nodes(nodes=[cmd.node], data=cmd.state)
                elif cmd.kind == CommandKind.E:
                    self.backend.entangle_nodes(edge=cmd.nodes)
                elif cmd.kind == CommandKind.M:
                    self.__measure_method.measure(self.backend, cmd)
                # Use of `==` here for mypy
                elif cmd.kind == CommandKind.X or cmd.kind == CommandKind.Z:  # noqa: PLR1714
                    self.backend.correct_byproduct(cmd, self.__measure_method)
                elif cmd.kind == CommandKind.C:
                    self.backend.apply_clifford(cmd.node, cmd.clifford)
                else:
                    raise ValueError("invalid commands")
            self.backend.finalize(output_nodes=self.pattern.output_nodes)
        else:
            self.noise_model.assign_simulator(self)
            for node in self.pattern.input_nodes:
                self.backend.apply_channel(self.noise_model.prepare_qubit(), [node])
            for cmd in self.pattern:
                if cmd.kind == CommandKind.N:
                    self.backend.add_nodes([cmd.node])
                    self.backend.apply_channel(self.noise_model.prepare_qubit(), [cmd.node])
                elif cmd.kind == CommandKind.E:
                    self.backend.entangle_nodes(cmd.nodes)
                    self.backend.apply_channel(self.noise_model.entangle(), cmd.nodes)
                elif cmd.kind == CommandKind.M:
                    self.backend.apply_channel(self.noise_model.measure(), [cmd.node])
                    self.__measure_method.measure(self.backend, cmd, noise_model=self.noise_model)
                elif cmd.kind == CommandKind.X:
                    self.backend.correct_byproduct(cmd, self.__measure_method)
                    if np.mod(sum(self.__measure_method.get_measure_result(j) for j in cmd.domain), 2) == 1:
                        self.backend.apply_channel(self.noise_model.byproduct_x(), [cmd.node])
                elif cmd.kind == CommandKind.Z:
                    self.backend.correct_byproduct(cmd, self.__measure_method)
                    if np.mod(sum(self.__measure_method.get_measure_result(j) for j in cmd.domain), 2) == 1:
                        self.backend.apply_channel(self.noise_model.byproduct_z(), [cmd.node])
                elif cmd.kind == CommandKind.C:
                    self.backend.apply_clifford(cmd.node, cmd.clifford)
                    self.backend.apply_channel(self.noise_model.clifford(), [cmd.node])
                elif cmd.kind == CommandKind.T:
                    # T command is a flag for one clock cycle in simulated experiment,
                    # to be added via hardware-agnostic pattern modifier
                    self.noise_model.tick_clock()
                else:
                    raise ValueError("Invalid commands.")
            self.backend.finalize(self.pattern.output_nodes)
