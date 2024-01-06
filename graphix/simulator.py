"""MBQC simulator

Simulates MBQC by executing the pattern.

"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

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
class JittablePatternCommands:
    """Jittable pattern command
    This class is used to make the pattern sequence jittable. This is necessary because
    `jax` does not allow arrays of different shapes to be used in a jitted function.

    Example:
    .. code-block:: python
        jpc = JittablePatternCommands(
            jnp.array([1, 1, 2]),
            jnp.array([1, 2, 2]),
            jnp.array([(1, 1), (2, 2), (1, 2)]),
            jnp.array([-1, -1, -1]),
            jnp.array([jnp.nan, jnp.nan, jnp.nan]),
            jnp.array([[False, False], [False, False], [False, False]]),
            jnp.array([[False, False], [False, False], [False, False]]),
            jnp.array([[False, False], [False, False], [False, False]]),
            jnp.array([-1, -1, -1]),
        )

    Parameters
    ----------
    name: int
        command name. 'N'==1, 'E'==2, 'M'==3, 'X'==4, 'Z'==5, 'C'==6.
    node: int
        node index. Used for 'N', 'M', 'X', 'Z', 'C' commands. For 'E' command, it is the first node index.
    edge: tuple[int, int]
        edge. Used for 'E' command. For other commands, it is (node, node).
    plane: int
        measurement plane. Used for 'M' command. 'XY'==0, 'YZ'==1, 'XZ'==2.
        For other commands, it is -1.
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

    name: int
    node: int
    edge: tuple[int, int]
    plane: int
    angle: float
    s_domain: jax.Array
    t_domain: jax.Array
    signal_domain: jax.Array
    vop: int


def _plane_to_jittable_plane(plane: str) -> int:
    if plane == "XY":
        return 0
    elif plane == "YZ":
        return 1
    elif plane == "XZ":
        return 2
    else:
        return -1


def _pattern_seq_to_jittable_pattern_seq(pattern: Pattern):  # TODO: jnp or np array to jnp array, which is faster?
    import jax.numpy as jnp

    name = jnp.array([], dtype=np.int8)
    node = jnp.array([], dtype=np.int64)
    edge = jnp.array([], dtype=np.int64)
    plane = jnp.array([], dtype=np.int8)
    angle = jnp.array([], dtype=np.float64)
    s_domain = jnp.array([], dtype=np.bool_)
    t_domain = jnp.array([], dtype=np.bool_)
    signal_domain = jnp.array([], dtype=np.bool_)
    vop = jnp.array([], dtype=np.int8)

    measurement_results = {}

    for cmd in pattern.seq:
        if cmd[0] == "N":
            name = jnp.append(name, 1)
            node = jnp.append(node, cmd[1])
            edge = jnp.append(edge, jnp.array([cmd[1], cmd[1]], dtype=np.int64))
            plane = jnp.append(plane, -1)
            angle = jnp.append(angle, jnp.nan)
            s_domain = jnp.append(s_domain, jnp.zeros(pattern.Nnode, dtype=np.bool_))
            t_domain = jnp.append(t_domain, jnp.zeros(pattern.Nnode, dtype=np.bool_))
            signal_domain = jnp.append(signal_domain, jnp.zeros(pattern.Nnode, dtype=np.bool_))
            vop = jnp.append(vop, -1)
        elif cmd[0] == "E":
            name = jnp.append(name, 2)
            node = jnp.append(node, cmd[1][0])
            edge = jnp.append(edge, jnp.array([cmd[1][0], cmd[1][1]], dtype=np.int64))
            plane = jnp.append(plane, -1)
            angle = jnp.append(angle, jnp.nan)
            s_domain = jnp.append(s_domain, jnp.zeros(pattern.Nnode, dtype=np.bool_))
            t_domain = jnp.append(t_domain, jnp.zeros(pattern.Nnode, dtype=np.bool_))
            signal_domain = jnp.append(signal_domain, jnp.zeros(pattern.Nnode, dtype=np.bool_))
            vop = jnp.append(vop, -1)
        elif cmd[0] == "M":
            result = sim_backend.random_choice(sim_backend.array([False, True]))
            measurement_results[cmd[1]] = result

            name = jnp.append(name, 3)
            node = jnp.append(node, cmd[1])
            edge = jnp.append(edge, jnp.array([cmd[1], cmd[1]], dtype=np.int64))
            plane = jnp.append(plane, _plane_to_jittable_plane(cmd[2]))
            angle = jnp.append(angle, cmd[3])
            s_domain = jnp.append(
                s_domain, sim_backend.array([measurement_results.get(j, False) for j in cmd[4]], dtype=np.bool_)
            )
            t_domain = jnp.append(
                t_domain, sim_backend.array([measurement_results.get(j, False) for j in cmd[5]], dtype=np.bool_)
            )
            signal_domain = jnp.append(signal_domain, jnp.zeros(pattern.Nnode, dtype=np.bool_))
            vop = jnp.append(vop, -1)
        elif cmd[0] == "X":
            name = jnp.append(name, 4)
            node = jnp.append(node, cmd[1])
            edge = jnp.append(edge, jnp.array([cmd[1], cmd[1]], dtype=np.int64))
            plane = jnp.append(plane, -1)
            angle = jnp.append(angle, jnp.nan)
            s_domain = jnp.append(s_domain, jnp.zeros(pattern.Nnode, dtype=np.bool_))
            t_domain = jnp.append(t_domain, jnp.zeros(pattern.Nnode, dtype=np.bool_))
            signal_domain = jnp.append(signal_domain, jnp.zeros(pattern.Nnode, dtype=np.bool_))
            vop = jnp.append(vop, -1)
        elif cmd[0] == "Z":
            name = jnp.append(name, 5)
            node = jnp.append(node, cmd[1])
            edge = jnp.append(edge, jnp.array([cmd[1], cmd[1]], dtype=np.int64))
            plane = jnp.append(plane, -1)
            angle = jnp.append(angle, jnp.nan)
            s_domain = jnp.append(s_domain, jnp.zeros(pattern.Nnode, dtype=np.bool_))
            t_domain = jnp.append(t_domain, jnp.zeros(pattern.Nnode, dtype=np.bool_))
            signal_domain = jnp.append(signal_domain, jnp.zeros(pattern.Nnode, dtype=np.bool_))
            vop = jnp.append(vop, -1)
        elif cmd[0] == "C":
            name = jnp.append(name, 6)
            node = jnp.append(node, cmd[1])
            edge = jnp.append(edge, jnp.array([cmd[1], cmd[1]], dtype=np.int64))
            plane = jnp.append(plane, -1)
            angle = jnp.append(angle, jnp.nan)
            s_domain = jnp.append(s_domain, jnp.zeros(pattern.Nnode, dtype=np.bool_))
            t_domain = jnp.append(t_domain, jnp.zeros(pattern.Nnode, dtype=np.bool_))
            signal_domain = jnp.append(signal_domain, jnp.zeros(pattern.Nnode, dtype=np.bool_))
            vop = jnp.append(vop, cmd[2])
        else:
            raise ValueError("invalid command: {}".format(cmd))

    return JittablePatternCommands(name, node, edge, plane, angle, s_domain, t_domain, signal_domain, vop)


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
