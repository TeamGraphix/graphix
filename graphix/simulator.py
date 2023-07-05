"""MBQC simulator

Simulates MBQC by executing the pattern.

"""

from graphix.sim.tensornet import TensorNetworkBackend
from graphix.sim.statevec import StatevectorBackend
from graphix.sim.density_matrix import DensityMatrixBackend
from graphix.noise_models import BaseNoiseModel


class PatternSimulator:
    """MBQC simulator

    Executes the measurement pattern.
    """

    def __init__(self, pattern, backend="statevector", **kwargs):
        """
        Parameters
        -----------
        pattern: :class:`graphix.pattern.Pattern` object
            MBQC pattern to be simulated.
        backend: str, 'statevector' or 'tensornetwork'
            simulation backend (optional), default is 'statevector'.
        kwargs: keyword args for specified backend.

        .. seealso:: :class:`graphix.sim.statevec.StatevectorBackend`\
            :class:`graphix.sim.tensornet.TensorNetworkBackend`
        """
        # check that pattern has output nodes configured
        assert len(pattern.output_nodes) > 0
        if backend == "statevector":
            self.backend = StatevectorBackend(pattern, **kwargs)
        elif backend in {"tensornetwork", "mps"}:
            self.backend = TensorNetworkBackend(pattern, **kwargs)
        else:
            raise ValueError("unknown backend")
        self.pattern = pattern
        self.results = self.backend.results
        self.state = self.backend.state
        self.node_index = []

    def run(self):
        """Perform the simulation.

        Returns
        -------
        state :
            the output quantum state,
            in the representation depending on the backend used.
        """
        self.backend.add_nodes(self.pattern.input_nodes)
        for cmd in self.pattern:
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
            set_noise_model(self, BaseNoiseModel)
        else:
            set_noise_model(self, noise_model)

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
