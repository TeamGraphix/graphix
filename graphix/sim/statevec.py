from copy import deepcopy
import numbers
import typing

import numpy as np
import functools
import pydantic
import warnings

from graphix.clifford import CLIFFORD, CLIFFORD_CONJ, CLIFFORD_MUL
from graphix.states import State
from graphix.ops import Ops
import graphix.sim.base_backend
from graphix.sim.base_backend import BackendState
import graphix.states
import graphix.pauli
import graphix.types

# Python >= 3.9
# from collections.abc import Iterable # or use Protocols?
# https://stackoverflow.com/questions/49427944/typehints-for-sized-iterable-in-python
# Python >= 3.8
# typing.Iterable[T]

class StatevectorBackend(graphix.sim.base_backend.Backend):
    """MBQC simulator with statevector method."""

    def __init__(self, max_qubit_num=20, pr_calc=True):
        self.max_qubit_num = max_qubit_num
        super().__init__(pr_calc)
        self.to_trace = []
        self.to_trace_loc = []
        # Modify this
        self.results = {}

    def prepare_state(self, nodes, data) :
        self.add_nodes(nodes=nodes, input_state=Statevec(data))

    ## TODO : delete ?
    # def qubit_dim(self):
    #     """Returns the qubit number in the internal statevector

    #     Returns
    #     -------
    #     n_qubit : int
    #     """
    #     return len(self.state.dims())

    def add_nodes(self, backendState:BackendState, nodes, data=graphix.states.BasicStates.PLUS):
        """add new qubit(s) to statevector in argument
        and assign the corresponding node number
        to list self.node_index.

        Parameters
        ----------
        nodes : list of node indices
        """
        if not backendState.state:
            backendState.state = Statevec(nqubit=0)
        n = len(nodes)
        sv_to_add = Statevec(nqubit=n, data=data)

        backendState.state.tensor(sv_to_add)
        backendState.node_index.extend(nodes)
        
        return backendState
    
    def entangle_nodes(self, backendState:BackendState, edge):
        """Apply CZ gate to two connected nodes

        Parameters
        ----------
        edge : tuple (i, j)
            a pair of node indices
        """
        target = backendState.node_index.index(edge[0])
        control = backendState.node_index.index(edge[1])
        backendState.state.entangle((target, control))
        return backendState

    def measure(self, backendState:BackendState, node, measurement_description):
        """Perform measurement of a node in the internal statevector and trace out the qubit

        Parameters
        ----------
        cmd : list
            measurement command : ['M', node, plane, angle, s_domain, t_domain]
        """
        loc, result = self._perform_measure(backendState=backendState, node=node, measurement_description=measurement_description)
        backendState.state.remove_qubit(loc)
        return backendState, result

    def correct_byproduct(self, backendState:BackendState, results, cmd):
        """Byproduct correction
        correct for the X or Z byproduct operators,
        by applying the X or Z gate.
        """
        if np.mod(np.sum([results[j] for j in cmd[2]]), 2) == 1:
            if cmd[0] == "X":
                op = Ops.x
            elif cmd[0] == "Z":
                op = Ops.z
            self.apply_single(backendState=backendState, node=cmd[1], op=op)
        return backendState
    
    def apply_single(self, backendState:BackendState, node, op) :
        index = backendState.node_index.index(node)
        backendState.state.evolve_single(op=op, i=index)
        return backendState

    def apply_clifford(self, backendState:BackendState, cmd):
        """Apply single-qubit Clifford gate,
        specified by vop index specified in graphix.clifford.CLIFFORD
        """
        loc = backendState.node_index.index(cmd[1])
        backendState.state.evolve_single(CLIFFORD[cmd[2]], loc)
        return backendState

    def finalize(self, backendState:BackendState, output_nodes):
        """to be run at the end of pattern simulation."""
        self.sort_qubits(backendState, output_nodes)
        backendState.state.normalize()
        return backendState

    def sort_qubits(self, backendState:BackendState, output_nodes):
        """sort the qubit order in internal statevector"""
        for i, ind in enumerate(output_nodes):
            if not backendState.node_index[i] == ind:
                move_from = backendState.node_index.index(ind)
                backendState.state.swap((i, move_from))
                backendState.node_index[i], backendState.node_index[move_from] = (
                    backendState.node_index[move_from],
                    backendState.node_index[i],
                )


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

    # TODO at this stage no need for indices just be careful of the ordering in add_nodes
    def __init__(
        self,
        data: typing.Union[
            graphix.states.State, "Statevec", typing.Iterable[graphix.states.State], typing.Iterable[numbers.Number]
        ] = graphix.states.BasicStates.PLUS,
        nqubit: typing.Optional[graphix.types.PositiveInt] = None,
    ):

        """Initialize statevector

        Parameters
        ----------
        data : is either
            - a single state (:class:`graphix.states.State` object). THen prepares all nodes in that state (tensor product)
            - a dictionary mapping the inputs to a :class:`graphix.states.State` object
            - an arbitrary :class:`graphix.statevec.Statevec` object (arbitrary input) # TODO work on that since just copy?
        nqubit : int, optional: ignored if iterable passed (State, direct data)
            number of qubits. Defaults to 1.
        # plus_states : bool, optional
            whether or not to start all qubits in + state or 0 state. Defaults to +

        Defaults to |+> states and 1 qubit.
        If nqubit > 1 and only one state : tensor all of them. Use the tensor method instead of hard code.
        """
        pydantic.TypeAdapter(typing.Optional[graphix.types.PositiveInt]).validate_python(nqubit)

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
        elif isinstance(data, typing.Iterable):
            input_list = list(data)
        else:
            raise TypeError(f"Incorrect type for data: {type(data)}")

        if len(input_list) == 0:
            if nqubit is not None and nqubit != 0:
                raise ValueError("nqubit is not null but input state is empty.")

            # warnings.warn(f"Called Statevec with 0 qubits. Ignoring the state.")
            self.psi = np.array(1, dtype=np.complex128)
        else:
            if isinstance(input_list[0], graphix.states.State):
                graphix.types.check_list_elements(input_list, graphix.states.State)
                if nqubit is None:
                    nqubit = len(input_list)
                elif nqubit != len(input_list):
                    raise ValueError("Mismatch between nqubit and length of input state")
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
                # just reshape
                # NOTE too many conversions to numpy arrays?
                self.psi = psi.reshape((2,) * nqubit)
            else:
                raise TypeError(
                    f"First element of data has type {type(input_list[0])} whereas Number or State is expected"
                )

    def __repr__(self):
        return f"Statevec object with statevector {self.psi} and length {self.dims()}."

    def evolve_single(self, op, i):
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
        assert not np.isclose(_get_statevec_norm(self.psi), 0)
        psi = self.psi.take(indices=0, axis=qarg)
        self.psi = psi if not np.isclose(_get_statevec_norm(psi), 0) else self.psi.take(indices=1, axis=qarg)
        self.normalize()

    def entangle(self, edge):
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

        # NOTE on tensor form not vector
        # deprecated
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


def _get_statevec_norm(psi):
    """returns norm of the state"""
    return np.sqrt(np.sum(psi.flatten().conj() * psi.flatten()))
