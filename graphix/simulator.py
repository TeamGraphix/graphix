"""MBQC simulator

Simulates MBQC by executing the pattern.

"""

import numpy as np

from graphix.sim.density_matrix import DensityMatrixBackend
from graphix.sim.statevec import StatevectorBackend
from graphix.sim.tensornet import TensorNetworkBackend
from graphix.noise_models import NoiseModel
from graphix.command import N, M, E, C, X, Z, T
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
                warnings.warn(
                    "Simulating using densitymatrix backend with no noise. To add noise to the simulation, give an object of `graphix.noise_models.Noisemodel` to `noise_model` keyword argument."
                )
            if noise_model is not None:
                self.set_noise_model(noise_model)
                # if noise: have to compute the probabilities
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

        self.backend.add_nodes(self.pattern.input_nodes)
        if self.noise_model is None:
            for cmd in self.pattern:
                match cmd:
                    case N(node=i):
                        self.backend.add_nodes([i])
                    case E(nodes=n):
                        self.backend.entangle_nodes(n)
                    case M(node=i, plane=p, angle=a, s_domain=s, t_domain=t, vop=v):
                        self.backend.measure(cmd)
                    case X(node=i, domain=d):
                        self.backend.correct_byproduct(cmd)
                    case Z(node=i, domain=d):
                        self.backend.correct_byproduct(cmd)
                    case C(node=i, cliff_index=c):
                        self.backend.apply_clifford(cmd)
                    case _:
                        raise ValueError("invalid commands")
            self.backend.finalize()
        else:
            self.noise_model.assign_simulator(self)
            for node in self.pattern.input_nodes:
                self.backend.apply_channel(self.noise_model.prepare_qubit(), [node])
            for cmd in self.pattern:
                match cmd:
                    case N(node=i):
                        self.backend.add_nodes([i])
                        self.backend.apply_channel(self.noise_model.prepare_qubit(), [i])
                    case E(nodes=e):
                        self.backend.entangle_nodes(e)  # for some reaon entangle doesn't get the whole command
                        self.backend.apply_channel(self.noise_model.entangle(), e)
                    case M(node=i, plane=p, angle=a, s_domain=s, t_domain=t, vop=v):
                        self.backend.apply_channel(self.noise_model.measure(), [i])
                        self.backend.measure(cmd)
                        self.noise_model.confuse_result(cmd)
                    case X(node=i, domain=d):
                        self.backend.correct_byproduct(cmd)
                        if np.mod(np.sum([self.results[j] for j in cmd.domain]), 2) == 1:
                            self.backend.apply_channel(self.noise_model.byproduct_x(), [cmd.node])
                    case Z(node=i, domain=d):
                        self.backend.correct_byproduct(cmd)
                        if np.mod(np.sum([self.results[j] for j in cmd.domain]), 2) == 1:
                            self.backend.apply_channel(self.noise_model.byproduct_z(), [cmd.node])
                    case C(node=i, cliff_index=c):
                        self.backend.apply_clifford(cmd)
                        self.backend.apply_channel(self.noise_model.clifford(), [cmd.node])
                    case T():
                        self.noise_model.tick_clock()
                    case _:
                        raise ValueError("Invalid commands.")
            self.backend.finalize()

        return self.backend.state
