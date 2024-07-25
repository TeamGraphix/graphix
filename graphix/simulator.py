"""MBQC simulator

Simulates MBQC by executing the pattern.

"""

from __future__ import annotations

import warnings

import numpy as np

from graphix.command import CommandKind
from graphix.sim.density_matrix import DensityMatrixBackend
from graphix.sim.statevec import StatevectorBackend
from graphix.sim.tensornet import TensorNetworkBackend


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
        # assert len(pattern.output_nodes) > 0

        if backend == "statevector" and noise_model is None:
            self.noise_model = None
            self.backend = StatevectorBackend(pattern, **kwargs)
        elif backend == "densitymatrix":
            if noise_model is None:
                self.noise_model = None
                self.backend = DensityMatrixBackend(pattern, **kwargs)
                warnings.warn(
                    "Simulating using densitymatrix backend with no noise. To add noise to the simulation, give an object of `graphix.noise_models.Noisemodel` to `noise_model` keyword argument.",
                    stacklevel=1,
                )
            if noise_model is not None:
                self.set_noise_model(noise_model)
                self.backend = DensityMatrixBackend(pattern, pr_calc=True, **kwargs)
        elif backend in {"tensornetwork", "mps"} and noise_model is None:
            self.noise_model = None
            self.backend = TensorNetworkBackend(pattern, **kwargs)
        # TODO or just do the noiseless sim with a warning?
        elif backend in {"statevector", "tensornetwork", "mps"} and noise_model is not None:
            raise ValueError(f"The backend {backend} doesn't support noise but noisemodel was provided.")
        else:
            raise ValueError("Unknown backend.")
        self.pattern = pattern
        self.node_index = []

    def set_noise_model(self, model):
        self.noise_model = model

    @property
    def results(self):
        return self.backend.results

    @property
    def state(self):
        return self.backend.state

    def run(self):
        """Perform the simulation.

        Returns
        -------
        state :
            the output quantum state,
            in the representation depending on the backend used.
        """
        if self.noise_model is None:
            for cmd in self.pattern:
                kind = cmd.kind
                if kind == CommandKind.N:
                    self.backend.add_nodes([cmd.node])
                elif kind == CommandKind.E:
                    self.backend.entangle_nodes(cmd.nodes)
                elif kind == CommandKind.M:
                    self.backend.measure(cmd)
                elif kind == CommandKind.X:
                    self.backend.correct_byproduct(cmd)
                elif kind == CommandKind.Z:
                    self.backend.correct_byproduct(cmd)
                elif kind == CommandKind.C:
                    self.backend.apply_clifford(cmd)
                else:
                    raise ValueError("invalid commands")
            self.backend.finalize()
        else:
            self.noise_model.assign_simulator(self)
            for node in self.pattern.input_nodes:
                self.backend.apply_channel(self.noise_model.prepare_qubit(), [node])
            for cmd in self.pattern:
                kind = cmd.kind
                if kind == CommandKind.N:
                    self.backend.add_nodes([cmd.node])
                    self.backend.apply_channel(self.noise_model.prepare_qubit(), [cmd.node])
                elif kind == CommandKind.E:
                    self.backend.entangle_nodes(cmd.nodes)  # for some reaon entangle doesn't get the whole command
                    self.backend.apply_channel(self.noise_model.entangle(), cmd.nodes)
                elif kind == CommandKind.M:
                    self.backend.apply_channel(self.noise_model.measure(), [cmd.node])
                    self.backend.measure(cmd)
                    self.noise_model.confuse_result(cmd)
                elif kind == CommandKind.X:
                    self.backend.correct_byproduct(cmd)
                    if np.mod(np.sum([self.results[j] for j in cmd.domain]), 2) == 1:
                        self.backend.apply_channel(self.noise_model.byproduct_x(), [cmd.node])
                elif kind == CommandKind.Z:
                    self.backend.correct_byproduct(cmd)
                    if np.mod(np.sum([self.results[j] for j in cmd.domain]), 2) == 1:
                        self.backend.apply_channel(self.noise_model.byproduct_z(), [cmd.node])
                elif kind == CommandKind.C:
                    self.backend.apply_clifford(cmd.node)
                    self.backend.apply_channel(self.noise_model.clifford(), [cmd.node])
                elif kind == CommandKind.T:
                    self.noise_model.tick_clock()
                else:
                    raise ValueError("Invalid commands.")
            self.backend.finalize()

        return self.backend.state
