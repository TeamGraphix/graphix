from __future__ import annotations

from copy import copy, deepcopy
from functools import partial
from typing import TYPE_CHECKING, Optional

import numpy as np

from graphix.clifford import CLIFFORD, CLIFFORD_CONJ, CLIFFORD_MUL
from graphix.ops import Ops, States

from .backends.backend_factory import backend

if TYPE_CHECKING:
    from graphix.pattern import Pattern


def _update_global_variables():
    """update global variables. numpy array cannot be used in jax.jit, so we need to convert to jax.numpy array"""
    global CLIFFORD, CLIFFORD_MUL, CLIFFORD_CONJ
    CLIFFORD = backend.array(CLIFFORD)
    CLIFFORD_MUL = backend.array(CLIFFORD_MUL)
    CLIFFORD_CONJ = backend.array(CLIFFORD_CONJ)
    global CZ_TENSOR, CNOT_TENSOR, SWAP_TENSOR
    CZ_TENSOR = backend.array(CZ_TENSOR)
    CNOT_TENSOR = backend.array(CNOT_TENSOR)
    SWAP_TENSOR = backend.array(SWAP_TENSOR)


class StatevectorBackend:
    """MBQC simulator with statevector method."""

    def __init__(self, pattern: Pattern, max_qubit_num: int = 20, seed: Optional[int] = None):
        """
        Parameters
        -----------
        pattern : :class:`graphix.pattern.Pattern` object
            MBQC pattern to be simulated.
        backend : str, 'statevector'
            optional argument for simulation.
        max_qubit_num : int
            optional argument specifying the maximum number of qubits
            to be stored in the statevector at a time.
            Defaults to 20.
        seed : int
            optional argument for random number generator.
        """
        # check that pattern has output nodes configured
        if not len(pattern.output_nodes) > 0:
            raise ValueError("Pattern.output_nodes is empty. Set output nodes and try again")
        if pattern.max_space() > max_qubit_num:
            raise ValueError("Pattern.max_space is larger than max_qubit_num. Increase max_qubit_num and try again")

        self.pattern = pattern
        self.results = deepcopy(pattern.results)
        self.node_index = []
        self.max_qubit_num = pattern.max_space()
        backend.set_random_state(seed)
        if backend.name == "jax":
            self.state = Statevec(nqubit=0, fixed_nqubit=self.max_qubit_num)
        else:
            self.state = None

        _update_global_variables()

    def qubit_dim(self):
        """Returns the qubit number in the internal statevector

        Returns
        -------
        n_qubit : int
        """
        return len(self.state.dims())

    # @backend.jit
    def add_nodes(self, nodes):
        """add new qubits to internal statevector
        and assign the corresponding node numbers
        to list self.node_index.

        Parameters
        ----------
        nodes : list of node indices
        """
        if backend.name == "numpy":
            if not self.state:
                self.state = Statevec(nqubit=0)
            sv_to_add = Statevec(nqubit=len(nodes))
            self.state.tensor(sv_to_add)
        else:
            self.state.add_qubits(len(nodes))

        self.node_index.extend(nodes)

    # @backend.jit
    def entangle_nodes(self, edge):
        """Apply CZ gate to two connected nodes

        Parameters
        ----------
        edge : tuple (i, j)
            a pair of node indices
        """
        target = self.node_index.index(edge[0])
        control = self.node_index.index(edge[1])
        self.state.entangle((target, control))

    # @backend.jit
    def measure(self, cmd):
        """Perform measurement of a node in the internal statevector and trace out the qubit

        Parameters
        ----------
        cmd : list
            measurement command : ['M', node, plane, angle, s_domain, t_domain]
        """
        # choose the measurement result randomly
        result = backend.random_choice(backend.array([0, 1], dtype=np.int32))
        self.results[cmd[1]] = result

        # extract signals for adaptive angle
        s_signal = backend.sum(backend.array([self.results[j] for j in cmd[4]], dtype=np.int32))
        t_signal = backend.sum(backend.array([self.results[j] for j in cmd[5]], dtype=np.int32))
        angle = cmd[3] * backend.pi
        if len(cmd) == 7:
            vop = cmd[6]
        else:
            vop = backend.array(0)
        vop = backend.where(backend.mod(s_signal, 2) == 1, CLIFFORD_MUL[1, vop], vop)
        vop = backend.where(backend.mod(t_signal, 2) == 1, CLIFFORD_MUL[3, vop], vop)
        err, m_op = backend.wrap_by_checkify(meas_op)(angle, vop=vop, plane=cmd[2], choice=result)
        if err is not None:
            err.throw()
        loc = self.node_index.index(cmd[1])
        self.state.evolve_single(m_op, loc)
        self.state.remove_qubit(loc)
        self.node_index.remove(cmd[1])

    # @backend.jit
    def correct_byproduct(self, cmd):
        """Byproduct correction
        correct for the X or Z byproduct operators,
        by applying the X or Z gate.
        """

        # original implementation:
        # if backend.mod(backend.sum(backend.array([self.results[j] for j in cmd[2]])), 2) == 1:
        #     loc = self.node_index.index(cmd[1])
        #     if cmd[0] == "X":
        #         op = Ops.x
        #     elif cmd[0] == "Z":
        #         op = Ops.z
        #     self.state.evolve_single(op, loc)

        def true_fun():
            loc = self.node_index.index(cmd[1])
            if cmd[0] == "X":
                op = Ops.x
            elif cmd[0] == "Z":
                op = Ops.z
            return self.state._evolve_single(self.state.psi, op, loc)

        predicate = backend.mod(backend.sum(backend.array([self.results[j] for j in cmd[2]])), 2) == 1
        self.state.psi = backend.cond(predicate, true_fun, lambda: self.state.psi)

    # @backend.jit
    def apply_clifford(self, cmd):
        """Apply single-qubit Clifford gate,
        specified by vop index specified in graphix.clifford.CLIFFORD
        """
        loc = self.node_index.index(cmd[1])
        self.state.evolve_single(CLIFFORD[cmd[2]], loc)

    # @backend.jit
    def finalize(self):
        """to be run at the end of pattern simulation."""
        self.sort_qubits()
        self.state.normalize()

    # @backend.jit
    def sort_qubits(self):
        """sort the qubit order in internal statevector"""
        for i, ind in enumerate(self.pattern.output_nodes):
            if not self.node_index[i] == ind:
                move_from = self.node_index.index(ind)
                self.state.swap((i, move_from))
                self.node_index[i], self.node_index[move_from] = (
                    self.node_index[move_from],
                    self.node_index[i],
                )

    # https://jax.readthedocs.io/en/latest/faq.html#strategy-3-making-customclass-a-pytree
    def _tree_flatten(self):
        children = (self.pattern, self.results, self.state, self.node_index)  # arrays / dynamic values
        aux_data = {"max_qubit_num": self.max_qubit_num}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


try:
    from jax import tree_util

    tree_util.register_pytree_node(
        StatevectorBackend, StatevectorBackend._tree_flatten, StatevectorBackend._tree_unflatten
    )
except ModuleNotFoundError:
    pass


@partial(backend.jit, static_argnums=(2))
def meas_op(angle, vop=0, plane="XY", choice=0):
    """Returns the projection operator for given measurement angle and local Clifford op (VOP).

    .. seealso:: :mod:`graphix.clifford`

    Parameters
    ----------
    angle : float
        original measurement angle in radian
    vop : int
        index of local Clifford (vop), see graphq.clifford.CLIFFORD
    plane : 'XY', 'YZ' or 'ZX'
        measurement plane on which angle shall be defined
    choice : 0 or 1
        choice of measurement outcome. measured eigenvalue would be (-1)**choice.

    Returns
    -------
    op : numpy array
        projection operator

    """
    backend.debug_assert_true(
        backend.logical_and(0 <= vop, vop < 24),
        "vop must be in range(24), but got {vop}",
        vop=backend.array(vop, dtype=np.int32),
    )
    backend.debug_assert_true(
        backend.logical_or(choice == 0, choice == 1),
        "choice must be 0 or 1, but got {choice}",
        choice=backend.array(choice, dtype=np.int32),
    )
    if plane == "XY":
        vec = (backend.cos(angle), backend.sin(angle), 0)
    elif plane == "YZ":
        vec = (0, backend.cos(angle), backend.sin(angle))
    elif plane == "XZ":
        vec = (backend.cos(angle), 0, backend.sin(angle))
    else:
        raise ValueError("plane must be 'XY', 'YZ' or 'ZX'")
    op_mat = backend.eye(2) / 2
    for i in range(3):
        op_mat += (-1) ** (choice) * vec[i] * CLIFFORD[i + 1] / 2
    op_mat = CLIFFORD[CLIFFORD_CONJ[vop]] @ op_mat @ CLIFFORD[vop]
    return op_mat


CZ_TENSOR = np.array(
    [[[[1, 0], [0, 0]], [[0, 1], [0, 0]]], [[[0, 0], [1, 0]], [[0, 0], [0, -1]]]],
    dtype=np.complex128,
)
CNOT_TENSOR = np.array(
    [[[[1, 0], [0, 0]], [[0, 1], [0, 0]]], [[[0, 0], [0, 1]], [[0, 0], [1, 0]]]],
    dtype=np.complex128,
)
SWAP_TENSOR = np.array(
    [[[[1, 0], [0, 0]], [[0, 0], [1, 0]]], [[[0, 1], [0, 0]], [[0, 0], [0, 1]]]],
    dtype=np.complex128,
)


class Statevec:
    """Simple statevector simulator"""

    def __init__(self, nqubit: int = 1, plus_states: bool = True, fixed_nqubit: Optional[int] = None):
        """Initialize statevector

        Parameters
        ----------
        nqubit : int, optional:
            number of qubits. Defaults to 1.
        plus_states : bool, optional
            whether or not to start all qubits in + state or 0 state. Defaults to +
        fixed_nqubit : int, optional
            if not None, the number of qubits will be fixed to this value.
        """
        if fixed_nqubit is not None:
            self.actual_nqubit = copy(nqubit)
            self.fixed_nqubit = copy(fixed_nqubit)
            nqubit = fixed_nqubit
        if plus_states:
            self.psi = backend.ones((2,) * nqubit) / 2 ** (nqubit / 2)
        else:
            self.psi = backend.zeros((2,) * nqubit)
            self.psi[(0,) * nqubit] = 1

    def __repr__(self):
        return f"Statevec, data={self.psi}, shape={self.dims()}"

    def add_qubits(self, nqubit: int):
        """add qubits to the system only if the number of qubits is fixed"""
        if hasattr(self, "actual_nqubit"):
            self.actual_nqubit += nqubit
        else:
            raise ValueError("number of qubits is not fixed")

    def evolve_single(self, op, i):
        """Single-qubit operation

        Parameters
        ----------
        op : numpy.ndarray
            2*2 matrix
        i : int
            qubit index
        """
        self.psi = self._evolve_single(self.psi, op, i)

    @staticmethod
    def _evolve_single(psi, op, i):
        """internal logic of `evolve_single`. This is to avoid leaking of self.psi in jax.jit"""
        psi = backend.tensordot(op, psi, (1, i))
        psi = backend.moveaxis(psi, 0, i)
        return psi

    def evolve(self, op, qargs):
        """Multi-qubit operation

        Parameters
        ----------
        op : numpy.ndarray
            2^n*2^n matrix
        qargs : list of int
            target qubits' indices
        """
        op_dim = int(np.log2(len(op)))
        shape = [2 for _ in range(2 * op_dim)]
        op_tensor = backend.reshape(op, shape)
        self.psi = backend.tensordot(
            op_tensor,
            self.psi,
            (tuple(op_dim + i for i in range(len(qargs))), tuple(qargs)),
        )
        self.psi = backend.moveaxis(self.psi, [i for i in range(len(qargs))], qargs)

    def dims(self):
        return self.psi.shape

    def ptrace(self, qargs):
        """Perform partial trace of the selected qubits.

        .. warning::
            This method currently assumes qubits in qargs to be separable from the rest
            (checks not implemented for speed).
            Otherwise, the state returned will be forced to be pure which will result in incorrect output.
            Correct behaviour will be implemented as soon as the densitymatrix class, currently under development
            (PR #64), is merged.

        Parameters
        ----------
        qargs : list of int
            qubit indices to trace over
        """
        nqubit_after = len(self.psi.shape) - len(qargs)
        psi = self.psi
        rho = backend.tensordot(psi, psi.conj(), axes=(qargs, qargs))  # density matrix
        rho = backend.reshape(rho, (2**nqubit_after, 2**nqubit_after))
        evals, evecs = backend.eigh(rho)  # back to statevector, density matrix is hermitian
        self.psi = backend.reshape(evecs[:, backend.argmax(evals)], (2,) * nqubit_after)

    def remove_qubit(self, qarg):
        r"""Remove a separable qubit from the system and assemble a statevector for remaining qubits.
        This results in the same result as partial trace, if the qubit `qarg` is separable from the rest.

        For a statevector :math:`\ket{\psi} = \sum c_i \ket{i}` with sum taken over
        :math:`i \in [ 0 \dots 00,\ 0\dots 01,\ \dots,\
        1 \dots 11 ]`, this method returns

        .. math::
            \begin{align}
                \ket{\psi}' =&
                    c_{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k}}0_{\mathrm{k+1}} \dots 00}
                    \ket{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k+1}} \dots 00} \\
                    & + c_{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k}}0_{\mathrm{k+1}} \dots 01}
                    \ket{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k+1}} \dots 01} \\
                    & + c_{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k}}0_{\mathrm{k+1}} \dots 10}
                    \ket{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k+1}} \dots 10} \\
                    & + \dots \\
                    & + c_{1 \dots 1_{\mathrm{k-1}}0_{\mathrm{k}}1_{\mathrm{k+1}} \dots 11}
                    \ket{1 \dots 1_{\mathrm{k-1}}1_{\mathrm{k+1}} \dots 11},
            \end{align}

        (after normalization) for :math:`k =` qarg. If the :math:`k` th qubit is in :math:`\ket{1}` state,
        above will return zero amplitudes; in such a case the returned state will be the one above with
        :math:`0_{\mathrm{k}}` replaced with :math:`1_{\mathrm{k}}` .

        .. warning::
            This method assumes the qubit with index `qarg` to be separable from the rest,
            and is implemented as a significantly faster alternative for partial trace to
            be used after single-qubit measurements.
            Care needs to be taken when using this method.
            Checks for separability will be implemented soon as an option.

        .. seealso::
            :meth:`graphix.sim.statevec.Statevec.ptrace` and warning therein.

        Parameters
        ----------
        qarg : int
            qubit index
        """
        # Error handling in jax is tricky
        # https://github.com/google/jax/issues/4257
        # TODO: use https://jax.readthedocs.io/en/latest/debugging/checkify_guide.html
        # FIXME:
        assert not np.isclose(_get_statevec_norm(self.psi), 0)
        # self.psi = backend.where(backend.isclose(_get_statevec_norm(self.psi), 0), backend.nan, self.psi)
        if not hasattr(self, "actual_nqubit"):
            psi = self.psi.take(indices=0, axis=qarg)
            self.psi = psi if not np.isclose(_get_statevec_norm(psi), 0) else self.psi.take(indices=1, axis=qarg)
            # self.psi = backend.where(
            #     backend.isclose(_get_statevec_norm(psi), 0), self.psi.take(indices=1, axis=qarg), psi
            # )
        else:
            self.actual_nqubit -= 1

            psi = self.psi.take(indices=0, axis=qarg)
            qubit_removed_psi = (
                psi if not np.isclose(_get_statevec_norm(psi), 0) else self.psi.take(indices=1, axis=qarg)
            )

            self.psi = backend.tensordot(qubit_removed_psi, backend.array(States.plus), axes=0)
        self.normalize()

        # assert not np.isclose(_get_statevec_norm(self.psi), 0)
        # psi = self.psi.take(indices=0, axis=qarg)
        # self.psi = psi if not np.isclose(_get_statevec_norm(psi), 0) else self.psi.take(indices=1, axis=qarg)
        # self.normalize()

    def entangle(self, edge):
        """connect graph nodes

        Parameters
        ----------
        edge : tuple of int
            (control, target) qubit indices
        """
        # contraction: 2nd index - control index, and 3rd index - target index.
        self.psi = backend.tensordot(CZ_TENSOR, self.psi, ((2, 3), edge))
        # sort back axes
        self.psi = backend.moveaxis(self.psi, (0, 1), edge)

    def tensor(self, other):
        r"""Tensor product state with other qubits.
        Results in self :math:`\otimes` other.

        Parameters
        ----------
        other : :class:`graphix.sim.statevec.Statevec`
            statevector to be tensored with self
        """
        psi_self = self.psi.flatten()
        psi_other = other.psi.flatten()
        total_num = len(self.dims()) + len(other.dims())
        self.psi = backend.kron(psi_self, psi_other).reshape((2,) * total_num)

    def CNOT(self, qubits):
        """apply CNOT

        Parameters
        ----------
        qubits : tuple of int
            (control, target) qubit indices
        """
        # contraction: 2nd index - control index, and 3rd index - target index.
        self.psi = backend.tensordot(CNOT_TENSOR, self.psi, ((2, 3), qubits))
        # sort back axes
        self.psi = backend.moveaxis(self.psi, (0, 1), qubits)

    def swap(self, qubits):
        """swap qubits

        Parameters
        ----------
        qubits : tuple of int
            (control, target) qubit indices
        """
        # contraction: 2nd index - control index, and 3rd index - target index.
        self.psi = backend.tensordot(SWAP_TENSOR, self.psi, ((2, 3), qubits))
        # sort back axes
        self.psi = backend.moveaxis(self.psi, (0, 1), qubits)

    def normalize(self):
        """normalize the state"""
        norm = _get_statevec_norm(self.psi)
        self.psi = self.psi / norm

    def flatten(self):
        """returns flattened statevector"""
        if not hasattr(self, "actual_nqubit"):
            return self.psi.flatten()
        else:
            arr = self.psi.copy()
            for _ in range(self.fixed_nqubit - self.actual_nqubit):
                arr = arr.take(indices=0, axis=-1)
            arr /= _get_statevec_norm(arr)
            return arr.flatten()

    def expectation_single(self, op, loc):
        """Expectation value of single-qubit operator.

        Parameters
        ----------
        op : numpy.ndarray
            2*2 operator
        loc : int
            target qubit index

        Returns
        -------
        complex : expectation value.
        """
        st1 = deepcopy(self)
        st1.normalize()
        st2 = st1.deepcopy(st1)
        st1.evolve_single(op, loc)
        return backend.dot(st2.psi.flatten().conjugate(), st1.psi.flatten())

    def expectation_value(self, op, qargs):
        """Expectation value of multi-qubit operator.

        Parameters
        ----------
        op : numpy.ndarray
            2^n*2^n operator
        qargs : list of int
            target qubit indices

        Returns
        -------
        complex : expectation value
        """
        st1 = deepcopy(self)
        st1.normalize()
        st2 = deepcopy(st1)
        st1.evolve(op, qargs)
        return backend.dot(st2.psi.flatten().conjugate(), st1.psi.flatten())

    # https://jax.readthedocs.io/en/latest/faq.html#strategy-3-making-customclass-a-pytree
    def _tree_flatten(self):
        children = (self.psi, self.actual_nqubit)  # arrays / dynamic values
        aux_data = {"fixed_nqubit": self.fixed_nqubit}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


try:
    from jax import tree_util

    tree_util.register_pytree_node(Statevec, Statevec._tree_flatten, Statevec._tree_unflatten)
except ModuleNotFoundError:
    pass


def _get_statevec_norm(psi):
    """returns norm of the state"""
    return backend.sqrt(backend.sum(psi.flatten().conj() * psi.flatten()))
