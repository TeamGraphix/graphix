from __future__ import annotations

import collections
import functools
import numbers
import sys
from copy import deepcopy

import numpy as np
import numpy.typing as npt

import graphix.parameter
import graphix.pauli
import graphix.sim.base_backend
import graphix.states
import graphix.types
from graphix import command
from graphix.clifford import CLIFFORD, CLIFFORD_CONJ
from graphix.ops import Ops


class StatevectorBackend(graphix.sim.base_backend.Backend):
    """MBQC simulator with statevector method."""

    def __init__(
        self,
        pattern,
        input_state: Data = graphix.states.BasicStates.PLUS,
        max_qubit_num=20,
        pr_calc=True,
        rng: np.random.Generator | None = None,
    ):
        """
        Parameters
        -----------
        pattern : :class:`graphix.pattern.Pattern` object
            MBQC pattern to be simulated.
        input_state: same syntax as `graphix.statevec.Statevec` constructor.
        max_qubit_num : int
            optional argument specifying the maximum number of qubits
            to be stored in the statevector at a time.
        pr_calc : bool
            whether or not to compute the probability distribution before choosing the measurement result.
            if False, measurements yield results 0/1 with 50% probabilities each.
        """
        # check that pattern has output nodes configured
        # assert len(pattern.output_nodes) > 0
        self.pattern = pattern
        if pattern._pauli_preprocessed and input_state != graphix.states.BasicStates.PLUS:
            raise NotImplementedError(
                "Pauli preprocessing is currently only available when inputs are initialized in |+> state (see https://github.com/TeamGraphix/graphix/issues/168 )."
            )
        self.results = deepcopy(pattern.results)
        self.state = None
        self.node_index = []
        self.Nqubit = 0
        self.to_trace = []
        self.to_trace_loc = []
        self.max_qubit_num = max_qubit_num
        if pattern.max_space() > max_qubit_num:
            raise ValueError("Pattern.max_space is larger than max_qubit_num. Increase max_qubit_num and try again.")
        super().__init__(pr_calc, rng)

        # initialize input qubits to desired init_state
        self.add_nodes(pattern.input_nodes, input_state)

    def qubit_dim(self):
        """Returns the qubit number in the internal statevector

        Returns
        -------
        n_qubit : int
        """
        return len(self.state.dims())

    def add_nodes(self, nodes: list[int], input_state=graphix.states.BasicStates.PLUS) -> None:
        """add new qubit to internal statevector
        and assign the corresponding node number
        to list self.node_index.

        Parameters
        ----------
        nodes : list of node indices
        """
        if not self.state:
            self.state = Statevec(nqubit=0)
        n = len(nodes)
        sv_to_add = Statevec(nqubit=n, data=input_state)
        self.state.tensor(sv_to_add)
        self.node_index.extend(nodes)
        self.Nqubit += n

    def entangle_nodes(self, edge: tuple[int]):
        """Apply CZ gate to two connected nodes

        Parameters
        ----------
        edge : tuple (i, j)
            a pair of node indices
        """
        target = self.node_index.index(edge[0])
        control = self.node_index.index(edge[1])
        self.state.entangle((target, control))

    def measure(self, cmd: command.M):
        """Perform measurement of a node in the internal statevector and trace out the qubit

        Parameters
        ----------
        cmd : list
            measurement command : ['M', node, plane angle, s_domain, t_domain]
        """
        loc = self._perform_measure(cmd)
        self.state.remove_qubit(loc)
        self.Nqubit -= 1

    def correct_byproduct(self, cmd: list[command.X, command.Z]):
        """Byproduct correction
        correct for the X or Z byproduct operators,
        by applying the X or Z gate.
        """
        if np.mod(np.sum([self.results[j] for j in cmd.domain]), 2) == 1:
            loc = self.node_index.index(cmd.node)
            op = Ops.x if isinstance(cmd, command.X) else Ops.z
            self.state.evolve_single(op, loc)

    def apply_clifford(self, cmd: command.C):
        """Apply single-qubit Clifford gate,
        specified by vop index specified in graphix.clifford.CLIFFORD
        """
        loc = self.node_index.index(cmd.node)
        self.state.evolve_single(CLIFFORD[cmd.cliff_index], loc)

    def finalize(self):
        """to be run at the end of pattern simulation."""
        self.sort_qubits()
        self.state.normalize()

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


# This function is no longer used
def meas_op(angle, vop=0, plane=graphix.pauli.Plane.XY, choice=0):
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
    assert vop in np.arange(24)
    assert choice in [0, 1]
    assert plane in [graphix.pauli.Plane.XY, graphix.pauli.Plane.YZ, graphix.pauli.Plane.XZ]
    if plane == graphix.pauli.Plane.XY:
        vec = (np.cos(angle), np.sin(angle), 0)
    elif plane == graphix.pauli.Plane.YZ:
        vec = (0, np.cos(angle), np.sin(angle))
    elif plane == graphix.pauli.Plane.XZ:
        vec = (np.cos(angle), 0, np.sin(angle))
    op_mat = np.eye(2, dtype=np.complex128) / 2
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
    """Statevector object"""

    def __init__(
        self,
        data: Data = graphix.states.BasicStates.PLUS,
        nqubit: graphix.types.PositiveOrNullInt | None = None,
    ):
        """Initialize statevector objects. The behaviour is as follows. `data` can be:
        - a single :class:`graphix.states.State` (classical description of a quantum state)
        - an iterable of :class:`graphix.states.State` objects
        - an iterable of scalars (A 2**n numerical statevector)
        - a `graphix.statevec.Statevec` object

        If `nqubit` is not provided, the number of qubit is inferred from `data` and checked for consistency.
        If only one :class:`graphix.states.State` is provided and nqubit is a valid integer, initialize the statevector
        in the tensor product state.
        If both `nqubit` and `data` are provided, consistency of the dimensions is checked.
        If a `graphix.statevec.Statevec` is passed, returns a copy.


        :param data: input data to prepare the state. Can be a classical description or a numerical input, defaults to graphix.states.BasicStates.PLUS
        :type data: Data, optional
        :param nqubit: number of qubits to prepare, defaults to None
        :type nqubit: int, optional
        """

        assert nqubit is None or isinstance(nqubit, numbers.Integral) and nqubit >= 0

        if isinstance(data, Statevec):
            # assert nqubit is None or len(state.flatten()) == 2**nqubit
            if nqubit is not None and len(data.flatten()) != 2**nqubit:
                raise ValueError(
                    f"Inconsistent parameters between nqubit = {nqubit} and the inferred number of qubit = {len(data.flatten())}."
                )
            self.psi = data.psi.copy()
            return

        if isinstance(data, graphix.states.State):
            if nqubit is None:
                nqubit = 1
            input_list = [data] * nqubit
        elif isinstance(data, collections.abc.Iterable):
            input_list = list(data)
        else:
            raise TypeError(f"Incorrect type for data: {type(data)}")

        if len(input_list) == 0:
            if nqubit is not None and nqubit != 0:
                raise ValueError("nqubit is not null but input state is empty.")

            self.psi = np.array(1, dtype=np.complex128)

        else:
            if isinstance(input_list[0], graphix.states.State):
                graphix.types.check_list_elements(input_list, graphix.states.State)
                if nqubit is None:
                    nqubit = len(input_list)
                elif nqubit != len(input_list):
                    raise ValueError("Mismatch between nqubit and length of input state.")
                list_of_sv = [s.get_statevector() for s in input_list]
                tmp_psi = functools.reduce(np.kron, list_of_sv)
                # reshape
                self.psi = tmp_psi.reshape((2,) * nqubit)
            elif isinstance(input_list[0], numbers.Number):
                graphix.types.check_list_elements(input_list, numbers.Number)
                if nqubit is None:
                    length = len(input_list)
                    if length & (length - 1):
                        raise ValueError("Length is not a power of two")
                    nqubit = length.bit_length() - 1
                elif nqubit != len(input_list).bit_length() - 1:
                    raise ValueError("Mismatch between nqubit and length of input state")
                psi = np.array(input_list)
                if not np.allclose(np.sqrt(np.sum(np.abs(psi) ** 2)), 1):
                    raise ValueError("Input state is not normalized")
                self.psi = psi.reshape((2,) * nqubit)
            else:
                raise TypeError(
                    f"First element of data has type {type(input_list[0])} whereas Number or State is expected"
                )

    def __repr__(self):
        return f"Statevec object with statevector {self.psi} and length {self.dims()}."

    def evolve_single(self, op: npt.NDArray, i: int):
        """Single-qubit operation

        Parameters
        ----------
        op : numpy.ndarray
            2*2 matrix
        i : int
            qubit index
        """
        self.psi = np.tensordot(op, self.psi, (1, i))
        self.psi = np.moveaxis(self.psi, 0, i)

    def evolve(self, op: np.ndarray, qargs: list[int]):
        """Multi-qubit operation

        Parameters
        ----------
        op : numpy.ndarray
            2^n*2^n matrix
        qargs : list of int
            target qubits' indices
        """
        op_dim = int(np.log2(len(op)))
        # TODO shape = (2,)* 2 * op_dim
        shape = [2 for _ in range(2 * op_dim)]
        op_tensor = op.reshape(shape)
        self.psi = np.tensordot(
            op_tensor,
            self.psi,
            (tuple(op_dim + i for i in range(len(qargs))), tuple(qargs)),
        )
        self.psi = np.moveaxis(self.psi, [i for i in range(len(qargs))], qargs)

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
        rho = np.tensordot(psi, psi.conj(), axes=(qargs, qargs))  # density matrix
        rho = np.reshape(rho, (2**nqubit_after, 2**nqubit_after))
        evals, evecs = np.linalg.eig(rho)  # back to statevector
        # NOTE works since only one 1 in the eigenvalues corresponding to the state
        # TODO use np.eigh since rho is Hermitian?
        self.psi = np.reshape(evecs[:, np.argmax(evals)], (2,) * nqubit_after)

    def remove_qubit(self, qarg: int):
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
        norm = _get_statevec_norm(self.psi)
        if isinstance(norm, numbers.Number):
            assert not np.isclose(norm, 0)
        psi = self.psi.take(indices=0, axis=qarg)
        norm = _get_statevec_norm(psi)
        self.psi = (
            psi
            if not isinstance(norm, numbers.Number) or not np.isclose(norm, 0)
            else self.psi.take(indices=1, axis=qarg)
        )
        self.normalize()

    def entangle(self, edge: tuple[int, int]):
        """connect graph nodes

        Parameters
        ----------
        edge : tuple of int
            (control, target) qubit indices
        """
        # contraction: 2nd index - control index, and 3rd index - target index.
        self.psi = np.tensordot(CZ_TENSOR, self.psi, ((2, 3), edge))
        # sort back axes
        self.psi = np.moveaxis(self.psi, (0, 1), edge)

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
        self.psi = np.kron(psi_self, psi_other).reshape((2,) * total_num)

    def CNOT(self, qubits):
        """apply CNOT

        Parameters
        ----------
        qubits : tuple of int
            (control, target) qubit indices
        """
        # contraction: 2nd index - control index, and 3rd index - target index.
        self.psi = np.tensordot(CNOT_TENSOR, self.psi, ((2, 3), qubits))
        # sort back axes
        self.psi = np.moveaxis(self.psi, (0, 1), qubits)

    def swap(self, qubits):
        """swap qubits

        Parameters
        ----------
        qubits : tuple of int
            (control, target) qubit indices
        """
        # contraction: 2nd index - control index, and 3rd index - target index.
        self.psi = np.tensordot(SWAP_TENSOR, self.psi, ((2, 3), qubits))
        # sort back axes
        self.psi = np.moveaxis(self.psi, (0, 1), qubits)

    def normalize(self):
        """normalize the state"""
        norm = _get_statevec_norm(self.psi)
        self.psi = self.psi / norm

    def flatten(self):
        """returns flattened statevector"""
        return self.psi.flatten()

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
        st2 = deepcopy(st1)
        st1.evolve_single(op, loc)
        return np.dot(st2.psi.flatten().conjugate(), st1.psi.flatten())

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
        return np.dot(st2.psi.flatten().conjugate(), st1.psi.flatten())

    def subs(self, variable, substitute) -> Statevec:
        """Return a copy of the state vector where all occurrences of
        the given variable in measurement angles are substituted by
        the given value.

        See https://github.com/TeamGraphix/graphix-symbolic for
        a symbolic parameter implementation that supports simulation.
        """
        result = Statevec()
        result.psi = np.vectorize(lambda value: graphix.parameter.subs(value, variable, substitute))(self.psi)
        return result


def _get_statevec_norm(psi):
    """returns norm of the state"""
    return np.sqrt(np.sum(psi.flatten().conj() * psi.flatten()))


if sys.version_info >= (3, 10):
    from collections.abc import Iterable

    Data = graphix.states.State | Statevec | Iterable[graphix.states.State] | Iterable[numbers.Number]
else:
    from typing import Iterable, Union

    Data = Union[
        graphix.states.State,
        Statevec,
        Iterable[graphix.states.State],
        Iterable[numbers.Number],
    ]
