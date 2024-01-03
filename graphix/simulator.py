"""MBQC simulator

Simulates MBQC by executing the pattern.

"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from graphix.sim.backends.backend_factory import backend as sim_backend
from graphix.sim.statevec import StatevectorBackend
from graphix.sim.tensornet import TensorNetworkBackend

if TYPE_CHECKING:
    from graphix.pattern import Pattern

try:
    import jax
    import jax_dataclasses as jdc
    from jax_dataclasses import pytree_dataclass
except ModuleNotFoundError:
    pytree_dataclass = lambda x: x


@pytree_dataclass
class JittablePatternCommand:
    """Jittable pattern command
    This class is used to make the pattern sequence jittable. This is necessary because
    `jax` does not allow arrays of different shapes to be used in a jitted function.

    Example:
    .. code-block:: python
        jpc = JittablePatternCommand(
            ["N", "N", "E"],
            jnp.array([1, 2, 2]),
            jnp.array([(1, 1), (2, 2), (1, 2)]),
            [None, None, None],
            jnp.array([jnp.nan, jnp.nan, jnp.nan]),
            jnp.array([[False, False], [False, False], [False, False]]),
            jnp.array([[False, False], [False, False], [False, False]]),
            jnp.array([[False, False], [False, False], [False, False]]),
            jnp.array([-1, -1, -1]),
        )

    Parameters
    ----------
    name: str
        command name
    node: int
        node index. Used for 'N', 'M', 'X', 'Z', 'C' commands. For 'E' command, it is the first node index.
    edge: tuple[int, int]
        edge. Used for 'E' command. For other commands, it is (node, node).
    plane: str
        measurement plane. Used for 'M' command. For other commands, it is None.
    angle: float
        measurement angle. Used for 'M' command. For other commands, it is `jnp.nan`.
    s_domain: jax.Array of bool
        s domain. Used for 'M' command. For other commands, it is `jnp.zeros(number_of_nodes)`.
        If the measurement result is 0, the s domain is `jnp.zeros(number_of_nodes)`.
        If the measurement result is 1, `s_domain[feedforward_domains] = True` where `feedforward_domains` is the
        feedforward domains of the measurement node and rest of the elements are `False`.
    t_domain: jax.Array of bool
        t domain. Used for 'M' command. For other commands, it is `jnp.zeros(number_of_nodes)`.
        If the measurement result is 0, the t domain is `jnp.zeros(number_of_nodes)`.
        If the measurement result is 1, `t_domain[feedforward_domains] = True` where `feedforward_domains` is the
        feedforward domains of the measurement node and rest of the elements are `False`.
    signal_domain: jax.Array of bool
        signal domain. Used for 'M' command. For other commands, it is `jnp.zeros(number_of_nodes)`.
        The definition of signal domain is `signal_domain[signal_domains] = meaurement_result_of_signal_domains`
        where `signal_domains` is the signal domains of the byproduct command and
        `meaurement_result_of_signal_domains` is the measurement result of the signal domains. The rest of the elements
        are `False`.
    vop: int
        value for clifford index. Used for 'C' command. It is an integer between 0 and 23.
        For other commands, it is `-1`.
    """

    name: jdc.Static[str]
    node: int
    edge: tuple[int, int]
    plane: jdc.Static[Optional[str]]
    angle: float
    s_domain: jax.Array
    t_domain: jax.Array
    signal_domain: jax.Array
    vop: int


def _pattern_seq_to_jittable_pattern_seq(pattern: Pattern):  # TODO:
    pass


class PatternSimulator:
    """MBQC simulator

    Executes the measurement pattern.
    """

    def __init__(self, pattern: Pattern, backend="statevector", **kwargs):
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
