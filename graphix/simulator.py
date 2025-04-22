"""MBQC simulator.

Simulates MBQC by executing the pattern.

"""

from __future__ import annotations

import abc
import warnings
from typing import TYPE_CHECKING

import numpy as np

from graphix.clifford import Clifford
from graphix.command import BaseM, CommandKind, M, MeasureUpdate
from graphix.measurements import Measurement
from graphix.sim.base_backend import Backend
from graphix.sim.density_matrix import DensityMatrixBackend
from graphix.sim.statevec import StatevectorBackend
from graphix.sim.tensornet import TensorNetworkBackend
from graphix.states import BasicStates

if TYPE_CHECKING:
    from graphix.pattern import Pattern


class MeasureMethod(abc.ABC):
    """Measure method used by the simulator, with default measurement method that implements MBQC.

    To be overwritten by custom measurement methods in the case of delegated QC protocols.

    Example: class `ClientMeasureMethod` in https://github.com/qat-inria/veriphix
    """

    def measure(self, backend: Backend, cmd) -> bool:
        """Perform a measure."""
        description = self.get_measurement_description(cmd)
        return backend.measure(cmd.node, description)

    @abc.abstractmethod
    def get_measurement_description(self, cmd: BaseM) -> Measurement:
        """Return the description of the measurement performed by a given measure command (possibly blind)."""
        ...

    @abc.abstractmethod
    def get_measure_result(self, node: int) -> bool:
        """Return the result of a previous measurement."""
        ...

    @abc.abstractmethod
    def set_measure_result(self, node: int, result: bool) -> None:
        """Store the result of a previous measurement."""
        ...


class DefaultMeasureMethod(MeasureMethod):
    """Default measurement method implementing standard measurement plane/angle update for MBQC."""

    def __init__(self, results=None):
        if results is None:
            results = {}
        self.results = results

    def get_measurement_description(self, cmd: BaseM) -> Measurement:
        """Return the description of the measurement performed by a given measure command (cannot be blind in the case of DefaultMeasureMethod)."""
        assert isinstance(cmd, M)
        angle = cmd.angle * np.pi
        # extract signals for adaptive angle
        s_signal = sum(self.results[j] for j in cmd.s_domain)
        t_signal = sum(self.results[j] for j in cmd.t_domain)
        measure_update = MeasureUpdate.compute(cmd.plane, s_signal % 2 == 1, t_signal % 2 == 1, Clifford.I)
        angle = angle * measure_update.coeff + measure_update.add_term
        return Measurement(angle, measure_update.new_plane)

    def get_measure_result(self, node: int) -> bool:
        """Return the result of a previous measurement."""
        return self.results[node]

    def set_measure_result(self, node: int, result: bool) -> None:
        """Store the result of a previous measurement."""
        self.results[node] = result


class PatternSimulator:
    """MBQC simulator.

    Executes the measurement pattern.
    """

    def __init__(
        self, pattern, backend="statevector", measure_method: MeasureMethod | None = None, noise_model=None, **kwargs
    ) -> None:
        """
        Construct a pattern simulator.

        Parameters
        ----------
        pattern: :class:`graphix.pattern.Pattern` object
            MBQC pattern to be simulated.
        backend: :class:`graphix.sim.backend.Backend` object,
            or 'statevector', or 'densitymatrix', or 'tensornetwork'
            simulation backend (optional), default is 'statevector'.
        noise_model:
        kwargs: keyword args for specified backend.

        .. seealso:: :class:`graphix.sim.statevec.StatevectorBackend`\
            :class:`graphix.sim.tensornet.TensorNetworkBackend`\
            :class:`graphix.sim.density_matrix.DensityMatrixBackend`\
        """
        if isinstance(backend, Backend):
            assert kwargs == {}
            self.backend = backend
        elif backend == "statevector":
            self.backend = StatevectorBackend(**kwargs)
        elif backend == "densitymatrix":
            if noise_model is None:
                self.backend = DensityMatrixBackend(**kwargs)
                warnings.warn(
                    "Simulating using densitymatrix backend with no noise. To add noise to the simulation, give an object of `graphix.noise_models.Noisemodel` to `noise_model` keyword argument.",
                    stacklevel=1,
                )
            else:
                self.backend = DensityMatrixBackend(pr_calc=True, **kwargs)
                self.set_noise_model(noise_model)
        elif backend in {"tensornetwork", "mps"}:
            self.backend = TensorNetworkBackend(pattern, **kwargs)
        else:
            raise ValueError("Unknown backend.")
        self.set_noise_model(noise_model)
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

    def set_noise_model(self, model):
        """Set a noise model."""
        self.noise_model = model

    def run(self, input_state=BasicStates.PLUS) -> None:
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
            pattern = self.pattern
        else:
            pattern = self.noise_model.input_nodes(self.pattern.input_nodes) if input_state is not None else []
            pattern.extend(self.noise_model.transpile(self.pattern))
        for cmd in pattern:
            if cmd.kind == CommandKind.N:
                self.backend.add_nodes(nodes=[cmd.node], data=cmd.state)
            elif cmd.kind == CommandKind.E:
                self.backend.entangle_nodes(edge=cmd.nodes)
            elif cmd.kind == CommandKind.M:
                result = self.__measure_method.measure(self.backend, cmd)
                if self.noise_model is not None:
                    result = self.noise_model.confuse_result(cmd, result)
                self.__measure_method.set_measure_result(cmd.node, result)
            # Use of `==` here for mypy
            elif cmd.kind == CommandKind.X or cmd.kind == CommandKind.Z:  # noqa: PLR1714
                self.backend.correct_byproduct(cmd, self.__measure_method)
            elif cmd.kind == CommandKind.C:
                self.backend.apply_clifford(cmd.node, cmd.clifford)
            elif cmd.kind == CommandKind.T:
                # T command is a flag for one clock cycle in simulated experiment,
                # to be added via hardware-agnostic pattern modifier
                if self.noise_model is not None:
                    self.noise_model.tick_clock()
            elif cmd.kind == CommandKind.A:
                self.backend.apply_noise(cmd.nodes, cmd.noise)
            else:
                raise ValueError("invalid commands")
        self.backend.finalize(output_nodes=self.pattern.output_nodes)
