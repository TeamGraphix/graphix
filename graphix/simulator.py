"""MBQC simulator

Simulates MBQC by executing the pattern.

"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from graphix.sim.backends.backend_factory import backend as sim_backend
from graphix.sim.statevec import JittableStatevectorBackend, StatevectorBackend
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
class JittablePatternSequence:
    """Jittable pattern command
    This class is used to make the pattern sequence jittable. This is necessary because
    `jax` does not allow arrays of different shapes to be used in a jitted function.

    Example:
    .. code-block:: python
        jpc = JittablePatternSequenceu(
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
        command name. 'N'==0, 'E'==1, 'M'==2, 'X'==3, 'Z'==4, 'C'==5.
    node: int
        node index. Used for 'N', 'M', 'X', 'Z', 'C' commands. For 'E' command, it is -1.
    edge: tuple[int, int]
        edge. Used for 'E' command. For other commands, it is (-1, -1).
    plane: int
        measurement plane. Used for 'M' command. 'XY'==0, 'YZ'==1, 'XZ'==2.
        For other commands, it is -1.
    angle: float
        measurement angle. Used for 'M' command. For other commands, it is `jnp.nan`.
    m_result: bool
        measurement result. Used for 'M' command. For other commands, it is `False`.
    s_signal: bool
        s domain. Used for 'M' command. For other commands, it is `False`.
    t_signal: bool
        t domain. Used for 'M' command. For other commands, it is `False`.
    b_signal: bool
        signal domain. Used for 'X' and 'Z' command. For other commands, it is `False`.
    vop: int
        value for clifford index. Used for 'C' command. It is an integer between 0 and 23.
        For other commands, it is `-1`.
    """

    name: int
    node: int
    edge: tuple[int, int]
    plane: int
    angle: float
    m_result: bool
    s_signal: bool
    t_signal: bool
    b_signal: bool
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


def _pattern_seq_to_jittable_pattern_seq(pattern: Pattern):  # FIXME: does not work with pauli measurements
    seq_length = len(pattern.seq)
    name = np.zeros(seq_length, dtype=np.int8)
    node = np.full(seq_length, -1, dtype=np.int64)
    edge = np.full((seq_length, 2), -1, dtype=np.int64)
    plane = np.full(seq_length, -1, dtype=np.int8)
    angle = np.full(seq_length, np.nan, dtype=np.float64)
    m_result = np.zeros(seq_length, dtype=np.bool_)
    s_signal = np.zeros(seq_length, dtype=np.bool_)
    t_signal = np.zeros(seq_length, dtype=np.bool_)
    b_signal = np.zeros(seq_length, dtype=np.bool_)
    vop = np.full(seq_length, -1, dtype=np.int8)

    measurement_results = {}

    for i, cmd in enumerate(pattern.seq):
        if cmd[0] == "N":
            name[i] = 0
            node[i] = cmd[1]
        elif cmd[0] == "E":
            name[i] = 1
            edge[i] = np.array([cmd[1][0], cmd[1][1]], dtype=np.int64)
        elif cmd[0] == "M":
            result = sim_backend.random_choice(sim_backend.array([False, True], dtype=np.bool_))
            measurement_results[cmd[1]] = result
            m_result[i] = result

            name[i] = 2
            node[i] = cmd[1]
            plane[i] = _plane_to_jittable_plane(cmd[2])
            angle[i] = cmd[3]
            for j in cmd[4]:
                s_signal[i] ^= measurement_results[j]
            for j in cmd[5]:
                t_signal[i] ^= measurement_results[j]
        elif cmd[0] == "X":
            name[i] = 3
            node[i] = cmd[1]
            for j in cmd[2]:
                b_signal[i] ^= measurement_results[j]
        elif cmd[0] == "Z":
            name[i] = 4
            node[i] = cmd[1]
            for j in cmd[2]:
                b_signal[i] ^= measurement_results[j]
        elif cmd[0] == "C":
            name[i] = 5
            node[i] = cmd[1]
            vop[i] = cmd[2]
        else:
            raise ValueError("invalid command: {}".format(cmd))

    return JittablePatternSequence(name, node, edge, plane, angle, m_result, s_signal, t_signal, b_signal, vop)


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
            if sim_backend.name == "numpy":
                self.pattern = pattern
                self.backend = StatevectorBackend(pattern, **kwargs)
            else:
                self.pattern = _pattern_seq_to_jittable_pattern_seq(pattern)
                self.backend = JittableStatevectorBackend(pattern, **kwargs)
        elif backend in {"tensornetwork", "mps"}:
            if sim_backend.name != "numpy":
                raise ValueError("tensornetwork backend only works with numpy backend")
            self.pattern = pattern
            self.backend = TensorNetworkBackend(pattern, **kwargs)
        else:
            raise ValueError("unknown backend")

    @sim_backend.jit
    def run(self):
        """Perform the simulation.

        Returns
        -------
        state :
            the output quantum state,
            in the representation depending on the backend used.
        """
        if isinstance(self.backend, JittableStatevectorBackend):

            def loop_func(carry, x):
                jax.lax.switch(  # FIXME: make backend agnostic
                    x.name,
                    [
                        lambda: self.backend.add_node(x.node),
                        lambda: self.backend.entangle_nodes(x.edge),
                        lambda: self.backend.measure(x.node, x.plane, x.angle, x.s_signal, x.t_signal, x.m_result),
                        lambda: self.backend.correct_byproduct(3, x.node, x.b_signal),
                        lambda: self.backend.correct_byproduct(4, x.node, x.b_signal),
                        lambda: self.backend.apply_clifford(x.node, x.vop),
                    ],
                )
                return self.backend.state, None

            state, _ = jax.lax.scan(loop_func, self.backend.state, self.pattern)
            self.backend.finalize()

            return self.backend.state

        else:
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
