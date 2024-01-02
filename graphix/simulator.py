"""MBQC simulator

Simulates MBQC by executing the pattern.

"""
from __future__ import annotations

from graphix.sim.backends.backend_factory import backend as sim_backend
from graphix.sim.statevec import StatevectorBackend
from graphix.sim.tensornet import TensorNetworkBackend


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
            if sim_backend.name != "numpy":
                raise ValueError("tensornetwork backend only works with numpy backend")
            self.backend = TensorNetworkBackend(pattern, **kwargs)
        else:
            raise ValueError("unknown backend")
        self.pattern = pattern

    # @sim_backend.jit
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
                raise ValueError("invalid command: {}".format(cmd))
            if self.pattern.seq[-1] == cmd:
                self.backend.finalize()

        return self.backend.state

        # TODO: make it jittable
        # def body_fun(i, state):
        #     cmd = self.pattern.seq[i]
        #     if cmd[0] == "N":
        #         self.backend.add_nodes([cmd[1]])
        #     elif cmd[0] == "E":
        #         self.backend.entangle_nodes(cmd[1])
        #     elif cmd[0] == "M":
        #         self.backend.measure(cmd)
        #     elif cmd[0] == "X":
        #         self.backend.correct_byproduct(cmd)
        #     elif cmd[0] == "Z":
        #         self.backend.correct_byproduct(cmd)
        #     elif cmd[0] == "C":
        #         self.backend.apply_clifford(cmd)
        #     else:
        #         raise ValueError("invalid command: {}".format(cmd))
        #     if self.pattern.seq[-1] == cmd:
        #         self.backend.finalize()
        #     return state

        # return sim_backend.fori_loop(0, len(self.pattern.seq), body_fun, None)

    # https://jax.readthedocs.io/en/latest/faq.html#strategy-3-making-customclass-a-pytree
    def _tree_flatten(self):
        children = ()  # arrays / dynamic values
        aux_data = {"pattern": self.pattern, "backend": self.backend}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


try:
    from jax import tree_util

    tree_util.register_pytree_node(PatternSimulator, PatternSimulator._tree_flatten, PatternSimulator._tree_unflatten)
except ModuleNotFoundError:
    pass
