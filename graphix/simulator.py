"""MBQC simulator

Simulates MBQC by executing the pattern.

"""

from __future__ import annotations

import abc
import warnings

import numpy as np

import graphix.clifford
from graphix.command import CommandKind
from graphix.pauli import MeasureUpdate, Plane
from graphix.sim.base_backend import Backend, MeasurementDescription
from graphix.sim.density_matrix import DensityMatrixBackend
from graphix.sim.statevec import StatevectorBackend
from graphix.sim.tensornet import TensorNetworkBackend


class MeasureMethod(abc.ABC):
    def measure(self, backend: Backend, cmd, noise_model=None) -> Backend:
        node = cmd[1]
        description = self.get_measurement_description(cmd)
        backend, result = backend.measure(node, description)
        if noise_model is not None:
            result = noise_model.confuse_result(result)
        self.set_measure_result(node, result)
        return backend

    @abc.abstractmethod
    def get_measurement_description(self, cmd) -> MeasurementDescription: ...

    @abc.abstractmethod
    def get_measure_result(self, node: int) -> bool: ...

    @abc.abstractmethod
    def set_measure_result(self, node: int, result: bool) -> None: ...


class DefaultMeasureMethod(MeasureMethod):
    def __init__(self, results=None):
        if results is None:
            results = dict()
        self.results = results

    def get_measurement_description(self, cmd) -> MeasurementDescription:
        angle = cmd[3] * np.pi
        # extract signals for adaptive angle
        s_signal = np.sum(self.results[j] for j in cmd[4])
        t_signal = np.sum(self.results[j] for j in cmd[5])
        if len(cmd) == 7:
            vop = cmd[6]
        else:
            vop = 0
        measure_update = MeasureUpdate.compute(
            Plane[cmd[2]], s_signal % 2 == 1, t_signal % 2 == 1, graphix.clifford.TABLE[vop]
        )
        angle = angle * measure_update.coeff + measure_update.add_term
        return MeasurementDescription(measure_update.new_plane, angle)

    def get_measure_result(self, node: int) -> bool:
        return self.results[node]

    def set_measure_result(self, node: int, result: bool) -> None:
        self.results[node] = result


class PatternSimulator:
    """MBQC simulator

    Executes the measurement pattern.
    """

    def __init__(
        self, pattern, backend="statevector", measure_method: MeasureMethod = None, noise_model=None, **kwargs
    ):
        """
        Parameters
        -----------
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
        # check that pattern has output nodes configured
        # assert len(pattern.output_nodes) > 0

        if isinstance(backend, Backend):
            assert kwargs == dict()
            self.backend = backend
        elif backend == "statevector":
            self.backend = StatevectorBackend(**kwargs)
        elif backend == "densitymatrix":
            if noise_model is None:
                self.noise_model = None
                self.backend = DensityMatrixBackend(**kwargs)
                warnings.warn(
                    "Simulating using densitymatrix backend with no noise. To add noise to the simulation, give an object of `graphix.noise_models.Noisemodel` to `noise_model` keyword argument."
                )
            else:
                self.backend = DensityMatrixBackend(pr_calc=True, **kwargs)
                self.set_noise_model(noise_model)
        elif backend in {"tensornetwork", "mps"} and noise_model is None:
            self.noise_model = None
            self.backend = TensorNetworkBackend(pattern, **kwargs)
        else:
            raise ValueError("Unknown backend.")
        self.set_noise_model(noise_model)
        self.pattern = pattern
        if measure_method is None:
            measure_method = DefaultMeasureMethod(pattern.results)
        self.__measure_method = measure_method

    @property
    def measure_method(self):
        return self.__measure_method

    def set_noise_model(self, model):
        if not isinstance(self.backend, DensityMatrixBackend) and model is not None:
            self.noise_model = None  # if not initialized yet
            raise ValueError(f"The backend {backend} doesn't support noise but noisemodel was provided.")
        self.noise_model = model

    def run(self, input_state=graphix.states.BasicStates.PLUS) -> Backend:
        """Perform the simulation.

        Returns
        -------
        state :
            the output quantum state,
            in the representation depending on the backend used.
        """
        backend = self.backend
        if input_state is not None:
            backend = backend.add_nodes(self.pattern.input_nodes, input_state)
        if self.noise_model is None:
            for cmd in self.pattern:
                if cmd[0] == CommandKind.N:
                    backend = backend.add_nodes(nodes=[cmd.node], data=cmd.state)
                elif cmd[0] == CommandKind.E:
                    backend = backend.entangle_nodes(edge=cmd.node)
                elif cmd[0] == CommandKind.M:
                    backend = self.__measure_method.measure(backend, cmd)
                elif cmd[0] == CommandKind.X:
                    backend = backend.correct_byproduct(cmd, self.__measure_method)
                elif cmd[0] == CommandKind.Z:
                    backend = backend.correct_byproduct(cmd, self.__measure_method)
                elif cmd[0] == CommandKind.C:
                    backend = backend.apply_clifford(cmd.clifford)
                else:
                    raise ValueError("invalid commands")
            backend = backend.finalize(output_nodes=self.pattern.output_nodes)
        else:
            self.noise_model.assign_simulator(self)
            for node in self.pattern.input_nodes:
                backend = backend.apply_channel(self.noise_model.prepare_qubit(), [node])
            for cmd in self.pattern:
                if cmd.kind == CommandKind.N:
                    backend = backend.add_nodes([cmd.node])
                    backend = backend.apply_channel(self.noise_model.prepare_qubit(), [cmd.node])
                elif cmd.kind == CommandKind.E:
                    backend = backend.entangle_nodes(cmd.nodes)
                    backend = backend.apply_channel(self.noise_model.entangle(), cmd.nodes)
                elif cmd.kind == CommandKind.M:
                    backend = backend.apply_channel(self.noise_model.measure(), [cmd.node])
                    backend = self.__measure_method.measure(backend, cmd, noise_model=self.noise_model)
                elif cmd.kind == CommandKind.X:
                    backend = backend.correct_byproduct(cmd, self.__measure_method)
                    if np.mod(np.sum([self.__measure_method.results[j] for j in cmd.domain]), 2) == 1:
                        backend = backend.apply_channel(self.noise_model.byproduct_x(), [cmd.node])
                elif cmd.kind == CommandKind.Z:
                    backend = backend.correct_byproduct(cmd, self.__measure_method)
                    if np.mod(np.sum([self.__measure_method.results[j] for j in cmd.domain]), 2) == 1:
                        backend = backend.apply_channel(self.noise_model.byproduct_z(), [cmd.node])
                elif cmd.kind == CommandKind.C:
                    backend = backend.apply_clifford(cmd.clifford)
                    backend = backend.apply_channel(self.noise_model.clifford(), [cmd.node])
                elif cmd.kind == CommandKind.T:
                    # T command is a flag for one clock cycle in simulated experiment,
                    # to be added via hardware-agnostic pattern modifier
                    self.noise_model.tick_clock()
                else:
                    raise ValueError("Invalid commands.")
            backend = backend.finalize(self.pattern.output_nodes)

        return backend
