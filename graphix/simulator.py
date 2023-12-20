"""MBQC simulator

Simulates MBQC by executing the pattern.

"""

import numpy as np

from graphix.sim.density_matrix import DensityMatrixBackend
from graphix.sim.statevec import StatevectorBackend
from graphix.sim.tensornet import TensorNetworkBackend
from graphix.noise_models import NoiseModel
import warnings


class PatternSimulator:
    """MBQC simulator

    Executes the measurement pattern.
    """

    def __init__(self, pattern, backend="statevector", noise_model=None, **kwargs):
        """
        Parameters
        -----------
        pattern: :class:`graphix.pattern.Pattern` object
            MBQC pattern to be simulated.
        backend: str, 'statevector', 'densitymatrix or 'tensornetwork'
            simulation backend (optional), default is 'statevector'.
        noise_model: 
        kwargs: keyword args for specified backend.

        .. seealso:: :class:`graphix.sim.statevec.StatevectorBackend`\
            :class:`graphix.sim.tensornet.TensorNetworkBackend`\
            :class:`graphix.sim.density_matrix.DensityMatrixBackend`\
        """
        # check that pattern has output nodes configured
        assert len(pattern.output_nodes) > 0

        if backend == "statevector" and noise_model is None:
            self.noise_model = None
            self.backend = StatevectorBackend(pattern, **kwargs)
        elif backend == "densitymatrix":
            if noise_model is None:
                self.noise_model = None
                # no noise: no need to compute probabilities
                self.backend = DensityMatrixBackend(pattern, **kwargs)
                warnings.warn("Simulating using densitymatrix backend with no noise. To add noise to the simulation, give an object of `graphix.noise_models.Noisemodel` to `noise_model` keyword argument ")
            if noise_model is not None:
                self.set_noise_model(noise_model)
                # if noise: have to compute the probabilities
                self.backend = DensityMatrixBackend(pattern, pr_calc=True, **kwargs)
        elif backend in {"tensornetwork", "mps"} and noise_model is None:
            self.noise_model = None
            self.backend = TensorNetworkBackend(pattern, **kwargs)
        # TODO or just do the noiseless sim with a warning?
        elif backend in {"statevector", "tensornetwork", "mps"} and noise_model is not None:
            raise ValueError(f"The backend {backend} doesn't support noise but noisemodel was provided")
        else:
            raise ValueError("Unknown backend.")
        self.pattern = pattern
        self.results = self.backend.results
        self.state = self.backend.state
        self.node_index = []

    def set_noise_model(self, model):
        self.noise_model = model

    def run(self):
        """Perform the simulation.

        Returns
        -------
        state :
            the output quantum state,
            in the representation depending on the backend used.
        """

        if self.noise_model is None:
            for cmd in self.pattern.seq:
                if cmd[0] == "N":
                    self.backend.add_nodes([cmd[1]])
                elif cmd[0] == "E":
                    self.backend.entangle_nodes(cmd[1])
                elif cmd[0] == "M":
                    self.backend.measure(cmd)
                elif cmd[0] == "X":
                    self.backend.correct_byproduct(cmd)
                elif cmd[0] == "Z":
                    self.backend.correct_byproduct(cmd)
                elif cmd[0] == "C":
                    self.backend.apply_clifford(cmd)
                else:
                    raise ValueError("invalid commands")
                if self.pattern.seq[-1] == cmd:
                    self.backend.finalize()
        else:
            self.noise_model.assign_simulator(self)
            for cmd in self.pattern.seq:
                if cmd[0] == "N":  # prepare clean qubit and apply channel
                    self.backend.add_nodes([cmd[1]])
                    self.backend.apply_channel(self.noise_model.prepare_qubit(), [cmd[1]])
                elif cmd[0] == "E":  # for "E" cmd[1] is already a tuyple
                    self.backend.entangle_nodes(cmd[1])  # for some reaon entangle doesn't get the whole command
                    self.backend.apply_channel(self.noise_model.entangle(), cmd[1])
                elif cmd[0] == "M":  # apply channel before measuring, then measur and confuse_result
                    self.backend.apply_channel(self.noise_model.measure(), [cmd[1]])
                    self.backend.measure(cmd)
                    self.noise_model.confuse_result(cmd)
                elif cmd[0] == "X":
                    self.backend.correct_byproduct(cmd)
                    if np.mod(np.sum([self.results[j] for j in cmd[2]]), 2) == 1:
                        self.backend.apply_channel(self.noise_model.byproduct_x(), [cmd[1]])
                elif cmd[0] == "Z":
                    self.backend.correct_byproduct(cmd)
                    if np.mod(np.sum([self.results[j] for j in cmd[2]]), 2) == 1:
                        self.backend.apply_channel(self.noise_model.byproduct_z(), [cmd[1]])
                elif cmd[0] == "C":
                    self.backend.apply_clifford(cmd)
                    self.backend.apply_channel(self.noise_model.clifford(), [cmd[1]])
                elif cmd[0] == "T":
                    # T command is a flag for one clock cycle in simulated experiment,
                    # to be added via hardware-agnostic pattern modifier
                    self.noise_model.tick_clock()
                else:
                    raise ValueError("Invalid commands.")
                if self.pattern.seq[-1] == cmd:
                    self.backend.finalize()

        return self.backend.state


class NoisyPatternSimulator:
    """MBQC simulator

    Executes the measurement pattern with densitymatrix backend.
    Noise models is specified by NoiseModel classes in graphix.noise_models module.
    """

    def __init__(self, pattern, noise_model, backend="densitymatrix", **kwargs):
        """
        Parameteres
        -----------
        pattern: :class:`graphix.pattern.Pattern` object
            MBQC pattern to be simulated.
        noise_model: :class:`graphix.noise_models.BaseNoiseModel`
        backend: str, 'densitymatrix'
            simulation backend (optional), default is 'densitymatrix'.
        kwargs: keyword args for specified backend.

        .. seealso:: :class:`graphix.sim.___`
        """
        # check that pattern has output nodes configured
        assert len(pattern.output_nodes) > 0
        if backend == "densitymatrix":
            self.backend = DensityMatrixBackend(pattern, **kwargs)
        else:
            raise ValueError("unknown backend")
        self.pattern = pattern
        self.results = self.backend.results
        self.state = self.backend.state
        self.node_index = []
        if noise_model is None:
            self.set_noise_model(self, NoiseModel)
        else:
            self.set_noise_model(self, noise_model)

    def set_noise_model(self, model):
        assert issubclass(model, NoiseModel)
        self.noise_model = model

    def run(self):
        """Perform the simulation.

        Returns
        -------
        state :
            the output quantum state,
            in the representation depending on the backend used.
        """
        # NOTE assign_simulator isn't supposed to be called with a simulator parameter as is base_noise_model.py?
        self.noise_model.assign_simulator(self)
        for cmd in self.pattern.seq:
            if cmd[0] == "N":
                new_ancilla = self.noise_model.prepare_qubit(cmd)
                self.backend.add_nodes([cmd[1]], qubit_to_add=new_ancilla)
            elif cmd[0] == "E":
                self.backend.entangle_nodes(cmd[1])
                self.noise_model.entangle(cmd)
            elif cmd[0] == "M":
                self.backend.measure(cmd)
                self.noise_model.measure(cmd)
                self.noise_model.confuse_result(cmd)
            elif cmd[0] == "X":
                self.backend.correct_byproduct(cmd)
                self.noise_model.byproduct_x(cmd)
            elif cmd[0] == "Z":
                self.backend.correct_byproduct(cmd)
                self.noise_model.byproduct_z(cmd)
            elif cmd[0] == "C":
                self.backend.apply_clifford(cmd)
                self.noise_model.clifford(cmd)
            elif cmd[0] == "T":
                # T command is a flag for one clock cycle in simulated experiment,
                # to be added via hardware-agnostic pattern modifier
                self.noise_model.tick_clock()
            else:
                raise ValueError("invalid commands")
            if self.pattern.seq[-1] == cmd:
                self.backend.finalize()

        return self.backend.state
