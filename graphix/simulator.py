"""MBQC simulator

Simulates MBQC by executing the pattern.

"""

import warnings

import numpy as np

from graphix.noise_models import NoiseModel
from graphix.sim.density_matrix import DensityMatrixBackend
from graphix.sim.statevec import StatevectorBackend
from graphix.sim.tensornet import TensorNetworkBackend

from graphix.sim.base_backend import Backend, State
from graphix.states import PlanarState


@dataclasses.dataclass
class MeasurementDescription:
    plane: graphix.pauli.Plane
    angle: float


class MeasureMethod(abc.ABC):
    @abc.abstractmethod
    def get_measurement_description(self, cmd) -> MeasurementDescription:
        ...

    @abc.abstractmethod
    def set_measure_result(self, cmd, result: bool) -> None:
        ...


class DefaultMeasureMethod(MeasureMethod):
    def get_measurement_description(self, cmd, results) -> MeasurementDescription:
        angle = cmd[3] * np.pi
        # extract signals for adaptive angle
        s_signal = np.sum(results[j] for j in cmd[4])
        t_signal = np.sum(results[j] for j in cmd[5])
        if len(cmd) == 7:
            vop = cmd[6]
        else:
            vop = 0
        measure_update = graphix.pauli.MeasureUpdate.compute(
            graphix.pauli.Plane[cmd[2]], s_signal % 2 == 1, t_signal % 2 == 1, graphix.clifford.TABLE[vop]
        )
        angle = angle * measure_update.coeff + measure_update.add_term
        return MeasurementDescription(measure_update.new_plane, angle)

    def set_measure_result(self, node, result: bool) -> None:
        pass


class PatternSimulator:
    """MBQC simulator

    Executes the measurement pattern.
    """

    def __init__(self, pattern, backend="statevector", results=dict(), noise_model=None, measure_method=None, **kwargs):
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

        if isinstance(backend, graphix.sim.base_backend.Backend):
            assert kwargs == dict()
            self.backend = backend
        elif backend == "statevector":
            self.backend = graphix.sim.statevec.StatevectorBackend(**kwargs)
        elif backend == "densitymatrix":
            self.backend = graphix.sim.density_matrix.DensityMatrixBackend(**kwargs)
            if noise_model is None:
                self.noise_model = None
                self.backend = DensityMatrixBackend(**kwargs)
                warnings.warn(
                    "Simulating using densitymatrix backend with no noise. To add noise to the simulation, give an object of `graphix.noise_models.Noisemodel` to `noise_model` keyword argument."
                )
            if noise_model is not None:
                self.set_noise_model(noise_model)
                self.backend = DensityMatrixBackend(pr_calc=True, **kwargs)
        elif backend in {"tensornetwork", "mps"} and noise_model is None:
            self.noise_model = None
            self.backend = TensorNetworkBackend(pattern, **kwargs)
        else:
            raise ValueError("Unknown backend.")
        self.set_noise_model(noise_model)
        self.pattern = pattern
        self.results = results
        if measure_method:
            self.measure_method = measure_method
        else:
            self.measure_method = DefaultMeasureMethod()

    def set_noise_model(self, model):
        if not isinstance(self.backend, graphix.sim.density_matrix.DensityMatrixBackend) and model is not None:
            self.noise_model = None  # if not initialized yet
            raise ValueError(f"The backend {backend} doesn't support noise but noisemodel was provided.")
        self.noise_model = model

    def run(self) -> Backend:
        """Perform the simulation.

        Returns
        -------
        state :
            the output quantum state,
            in the representation depending on the backend used.
        """
        backend = self.backend
        if self.noise_model is None:
            for cmd in self.pattern:
                if cmd[0] == "N":
                    if len(cmd) == 2:
                        backend = backend.add_nodes(nodes=[cmd[1]])
                    elif len(cmd) == 4:
                        backend = backend.add_nodes(
                            nodes=[cmd[1]], data=PlanarState(plane=cmd[2], angle=cmd[3] * np.pi / 4)
                        )
                elif cmd[0] == "E":
                    backend = backend.entangle_nodes(edge=cmd[1])
                elif cmd[0] == "M":
                    measurement_description = (
                        self.measure_method.get_measurement_description(cmd, self.results)
                        if not isinstance(self.backend, graphix.sim.tensornet.TensorNetworkBackend)
                        else cmd
                    )
                    backend, result = backend.measure(node=cmd[1], measurement_description=measurement_description)
                    self.results[cmd[1]] = result
                    self.measure_method.set_measure_result(node=cmd[1], result=result)
                elif cmd[0] == "X":
                    backend = backend.correct_byproduct(cmd=cmd, results=self.results)
                elif cmd[0] == "Z":
                    backend = backend.correct_byproduct(cmd=cmd, results=self.results)
                elif cmd[0] == "C":
                    backend = backend.apply_clifford(cmd=cmd)
                else:
                    raise ValueError("invalid commands")
            backend = backend.finalize(output_nodes=self.pattern.output_nodes)
        else:
            self.noise_model.assign_simulator(self)
            for node in self.pattern.input_nodes:
                backend = backend.apply_channel(self.noise_model.prepare_qubit(), [node])
            for cmd in self.pattern:
                if cmd[0] == "N":  # prepare clean qubit and apply channel
                    backend = backend.add_nodes([cmd[1]])
                    backend = backend.apply_channel(self.noise_model.prepare_qubit(), [cmd[1]])
                elif cmd[0] == "E":  # for "E" cmd[1] is already a tuyple
                    backend = backend.entangle_nodes(cmd[1])  # for some reaon entangle doesn't get the whole command
                    backend = backend.apply_channel(self.noise_model.entangle(), cmd[1])
                elif cmd[0] == "M":  # apply channel before measuring, then measur and confuse_result
                    measurement_description = (
                        self.measure_method.get_measurement_description(cmd, self.results)
                        if not isinstance(self.backend, graphix.sim.tensornet.TensorNetworkBackend)
                        else cmd
                    )
                    backend = backend.apply_channel(self.noise_model.measure(), [cmd[1]])
                    backend, result = backend.measure(node=cmd[1], measurement_description=measurement_description)
                    self.results[cmd[1]] = result
                    self.measure_method.set_measure_result(node=cmd[1], result=result)
                    self.noise_model.confuse_result(cmd)
                elif cmd[0] == "X":
                    backend = backend.correct_byproduct(results=self.results, cmd=cmd)
                    if np.mod(np.sum([self.results[j] for j in cmd[2]]), 2) == 1:
                        backend = backend.apply_channel(self.noise_model.byproduct_x(), [cmd[1]])
                elif cmd[0] == "Z":
                    backend = backend.correct_byproduct(results=self.results, cmd=cmd)
                    if np.mod(np.sum([self.results[j] for j in cmd[2]]), 2) == 1:
                        backend = backend.apply_channel(self.noise_model.byproduct_z(), [cmd[1]])
                elif cmd[0] == "C":
                    backend = backend.apply_clifford(cmd)
                    backend = backend.apply_channel(self.noise_model.clifford(), [cmd[1]])
                elif cmd[0] == "T":
                    # T command is a flag for one clock cycle in simulated experiment,
                    # to be added via hardware-agnostic pattern modifier
                    self.noise_model.tick_clock()
                else:
                    raise ValueError("Invalid commands.")
            backend = backend.finalize(self.pattern.output_nodes)

        return backend
