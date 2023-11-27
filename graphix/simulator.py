"""MBQC simulator

Simulates MBQC by executing the pattern.

"""

from graphix.sim.tensornet import TensorNetworkBackend
from graphix.sim.statevec import StatevectorBackend


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

        return self.backend.state
