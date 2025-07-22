"""Abstract base class for simulation backends."""

from __future__ import annotations

import dataclasses
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, SupportsFloat, TypeVar

import numpy as np
import numpy.typing as npt

# TypeAlias introduced in Python 3.10
# override introduced in Python 3.12
from typing_extensions import TypeAlias, override

from graphix.clifford import Clifford
from graphix.command import CommandKind
from graphix.measurements import outcome
from graphix.ops import Ops
from graphix.rng import ensure_rng
from graphix.states import BasicStates

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Iterator, Sequence

    from numpy.random import Generator

    from graphix import command
    from graphix.channels import KrausChannel
    from graphix.fundamentals import Plane
    from graphix.measurements import Measurement, Outcome
    from graphix.parameter import ExpressionOrComplex, ExpressionOrFloat
    from graphix.sim.data import Data
    from graphix.simulator import MeasureMethod


if sys.version_info >= (3, 10):
    Matrix: TypeAlias = npt.NDArray[np.object_ | np.complex128]
else:
    from typing import Union

    Matrix: TypeAlias = npt.NDArray[Union[np.object_, np.complex128]]


def tensordot(op: Matrix, psi: Matrix, axes: tuple[int | Sequence[int], int | Sequence[int]]) -> Matrix:
    """Tensor dot product that preserves the type of `psi`.

    This wrapper around `np.tensordot` ensures static type checking
    for both numeric (`complex128`) and symbolic (`object`) arrays.
    Even though the runtime behavior is the same, NumPy's static types don't
    support `Matrix` directly.

    If `psi` and `op` are numeric, the result is numeric.
    If `psi` or `op` are symbolic, the other is converted to symbolic if needed and
    the result is symbolic.

    Parameters
    ----------
    op : Matrix
        Operator tensor, either symbolic or numeric.
    psi : Matrix
        State tensor, either symbolic or numeric.
    axes : tuple[int | Sequence[int], int | Sequence[int]]
        Axes along which to contract `op` and `psi`.

    Returns
    -------
    Matrix
        The result of the tensor contraction with the same type as `psi`.
    """
    if psi.dtype == np.complex128 and op.dtype == np.complex128:
        psi_c = psi.astype(np.complex128, copy=False)
        op_c = op.astype(np.complex128, copy=False)
        return np.tensordot(op_c, psi_c, axes).astype(np.complex128)
    psi_o = psi.astype(np.object_, copy=False)
    op_o = op.astype(np.object_, copy=False)
    return np.tensordot(op_o, psi_o, axes)


def eig(mat: Matrix) -> tuple[Matrix, Matrix]:
    """Compute eigenvalues and eigenvectors of a matrix, preserving symbolic/numeric type.

    This wrapper around `np.linalg.eig` handles both numeric and symbolic matrices.
    Even though the runtime behavior is the same, NumPy's static types don't
    support `Matrix` directly.

    Parameters
    ----------
    mat : Matrix
        The matrix to diagonalize. Can be either `np.complex128` or `np.object_`.

    Returns
    -------
    tuple[Matrix, Matrix]
        A tuple `(w, v)` where `w` are the eigenvalues and `v` the right eigenvectors,
        with the same dtype as `mat`.

    Raises
    ------
    TypeError
        If `mat` has an unsupported dtype.
    """
    if mat.dtype == np.object_:
        mat_o = mat.astype(np.object_, copy=False)
        # mypy doesn't accept object dtype here
        return np.linalg.eig(mat_o)  # type: ignore[arg-type]

    mat_c = mat.astype(np.complex128, copy=False)
    return np.linalg.eig(mat_c)


def kron(a: Matrix, b: Matrix) -> Matrix:
    """Kronecker product with type-safe handling of symbolic and numeric matrices.

    The two matrices should have the same type.

    Parameters
    ----------
    a : Matrix
        Left operand (symbolic or numeric).
    b : Matrix
        Right operand (symbolic or numeric).

    Returns
    -------
    Matrix
        Kronecker product of `a` and `b`.

    Raises
    ------
    TypeError
        If `a` and `b` don't have the same type.
    """
    if a.dtype == np.complex128 and b.dtype == np.complex128:
        a_c = a.astype(np.complex128, copy=False)
        b_c = b.astype(np.complex128, copy=False)
        return np.kron(a_c, b_c).astype(np.complex128)

    if a.dtype == np.object_ and b.dtype == np.object_:
        a_o = a.astype(np.object_, copy=False)
        b_o = b.astype(np.object_, copy=False)
        return np.kron(a_o, b_o)

    raise TypeError("Operands should have the same type.")


def outer(a: Matrix, b: Matrix) -> Matrix:
    """Outer product with type-safe handling of symbolic and numeric vectors.

    The two matrices should have the same type.

    Parameters
    ----------
    a : Matrix
        Left operand (symbolic or numeric).
    b : Matrix
        Right operand (symbolic or numeric).

    Returns
    -------
    Matrix
        Outer product of `a` and `b`.

    Raises
    ------
    TypeError
        If `a` and `b` don't have the same type.
    """
    if a.dtype == np.complex128 and b.dtype == np.complex128:
        a_c = a.astype(np.complex128, copy=False)
        b_c = b.astype(np.complex128, copy=False)
        return np.outer(a_c, b_c).astype(np.complex128)

    if a.dtype == np.object_ and b.dtype == np.object_:
        a_o = a.astype(np.object_, copy=False)
        b_o = b.astype(np.object_, copy=False)
        return np.outer(a_o, b_o)

    raise TypeError("Operands should have the same type.")


def vdot(a: Matrix, b: Matrix) -> ExpressionOrComplex:
    """Conjugate dot product ⟨a|b⟩ with type-safe handling of symbolic and numeric vectors.

    The two matrices should have the same type.

    Parameters
    ----------
    a : Matrix
        Left operand (symbolic or numeric).
    b : Matrix
        Right operand (symbolic or numeric).

    Returns
    -------
    ExpressionOrFloat
        Dot product.

    Raises
    ------
    TypeError
        If `a` and `b` don't have the same type.
    """
    if a.dtype == np.complex128 and b.dtype == np.complex128:
        a_c = a.astype(np.complex128, copy=False)
        b_c = b.astype(np.complex128, copy=False)
        return complex(np.vdot(a_c, b_c))

    if a.dtype == np.object_ and b.dtype == np.object_:
        a_o = a.astype(np.object_, copy=False)
        b_o = b.astype(np.object_, copy=False)
        return np.vdot(a_o, b_o)  # type: ignore[no-any-return]

    raise TypeError("Operands should have the same type.")


def matmul(a: Matrix, b: Matrix) -> Matrix:
    """Matrix product a @ b with type-safe handling of symbolic and numeric vectors.

    The two matrices should have the same type.

    Parameters
    ----------
    a : Matrix
        Left operand (symbolic or numeric).
    b : Matrix
        Right operand (symbolic or numeric).

    Returns
    -------
    Matrix
        Matrix product.

    Raises
    ------
    TypeError
        If `a` and `b` don't have the same type.
    """
    if a.dtype == np.complex128 and b.dtype == np.complex128:
        a_c = a.astype(np.complex128, copy=False)
        b_c = b.astype(np.complex128, copy=False)
        return a_c @ b_c

    if a.dtype == np.object_ and b.dtype == np.object_:
        a_o = a.astype(np.object_, copy=False)
        b_o = b.astype(np.object_, copy=False)
        return a_o @ b_o  # type: ignore[no-any-return]

    raise TypeError("Operands should have the same type.")


class NodeIndex:
    """A class for managing the mapping between node numbers and qubit indices in the internal state of the backend.

    This allows for efficient access and manipulation of qubit orderings throughout the execution of a pattern.

    Attributes
    ----------
        __list (list): A private list of the current active node (labelled with integers).
        __dict (dict): A private dictionary mapping current node labels (integers) to their corresponding qubit indices
                       in the backend's internal quantum state.
    """

    __dict: dict[int, int]
    __list: list[int]

    def __init__(self) -> None:
        """Initialize an empty mapping between nodes and qubit indices."""
        self.__dict = {}
        self.__list = []

    def __getitem__(self, index: int) -> int:
        """Return the qubit node associated with the specified index.

        Parameters
        ----------
        index : int
            Position in the internal list.

        Returns
        -------
        int
            Node label corresponding to ``index``.
        """
        return self.__list[index]

    def index(self, node: int) -> int:
        """Return the qubit index associated with the specified node label.

        Parameters
        ----------
        node : int
            Node label to look up.

        Returns
        -------
        int
            Position of ``node`` in the internal ordering.
        """
        return self.__dict[node]

    def __iter__(self) -> Iterator[int]:
        """Return an iterator over node labels in their current order."""
        return iter(self.__list)

    def __len__(self) -> int:
        """Return the number of currently active nodes."""
        return len(self.__list)

    def extend(self, nodes: Iterable[int]) -> None:
        """Extend the mapping with additional nodes.

        Parameters
        ----------
        nodes : Iterable[int]
            Node labels to append.
        """
        base = len(self)
        self.__list.extend(nodes)
        # The following loop iterates over `self.__list[base:]` instead of `nodes`
        # because the iterable `nodes` can be transient and consumed by the
        # `self.__list.extend` on the line just above.
        for index, node in enumerate(self.__list[base:]):
            self.__dict[node] = base + index

    def remove(self, node: int) -> None:
        """Remove a node and reassign indices of the remaining nodes.

        Parameters
        ----------
        node : int
            Node label to remove.
        """
        index = self.__dict[node]
        del self.__list[index]
        del self.__dict[node]
        for new_index, u in enumerate(self.__list[index:], start=index):
            self.__dict[u] = new_index

    def swap(self, i: int, j: int) -> None:
        """Swap two nodes given their indices.

        Parameters
        ----------
        i, j : int
            Indices of the nodes in the current ordering.
        """
        node_i = self.__list[i]
        node_j = self.__list[j]
        self.__list[i] = node_j
        self.__list[j] = node_i
        self.__dict[node_i] = j
        self.__dict[node_j] = i


class NoiseNotSupportedError(Exception):
    """Exception raised when `apply_channel` is called on a backend that does not support noise."""

    def __str__(self) -> str:
        """Return the error message."""
        return "This backend does not support noise."


class BackendState(ABC):
    """
    Abstract base class for representing the quantum state of a backend.

    `BackendState` defines the interface for quantum state representations used by
    various backend implementations. It provides a common foundation for different
    simulation strategies, such as dense linear algebra or tensor network contraction.

    Concrete subclasses must implement the storage and manipulation logic appropriate
    for a specific backend and representation strategy.

    Notes
    -----
    This class is abstract and cannot be instantiated directly.

    Examples of concrete subclasses include:
    - :class:`Statevec` (for pure states represented as state vectors)
    - :class:`DensityMatrix` (for mixed states represented as density matrices)
    - :class:`MBQCTensorNet` (for compressed representations using tensor networks)

    See Also
    --------
    :class:`DenseState`, :class:`MBQCTensorNet`, :class:`Statevec`, :class:`DensityMatrix`
    """

    def apply_channel(self, channel: KrausChannel, qargs: Sequence[int]) -> None:
        """Apply channel to the state."""
        _ = self  # silence PLR6301
        _ = channel  # silence ARG002
        _ = qargs
        raise NoiseNotSupportedError

    @abstractmethod
    def flatten(self) -> Matrix:
        """Return flattened state."""


class DenseState(BackendState):
    """
    Abstract base class for quantum states with full dense representations.

    `DenseState` defines the shared interface and behavior for state representations
    that explicitly store the entire quantum state in memory as a dense array.
    This includes both state vectors (for pure states) and density matrices (for
    mixed states).

    This class serves as a common parent for :class:`Statevec` and :class:`DensityMatrix`, which
    implement the concrete representations of dense quantum states. It is used in
    simulation backends that operate using standard linear algebra on the full
    state, such as :class:`StatevecBackend` and :class:`DensityMatrixBackend`.

    Notes
    -----
    This class is abstract and cannot be instantiated directly.

    Not all :class:`BackendState` subclasses are dense. For example, :class:`MBQCTensorNet` is a
    `BackendState` that represents the quantum state using a tensor network, rather than
    a single dense array.

    See Also
    --------
    :class:`Statevec`, :class:`DensityMatrix`
    """

    # Note that `@property` must appear before `@abstractmethod` for pyright
    @property
    @abstractmethod
    def nqubit(self) -> int:
        """Return the number of qubits."""

    @abstractmethod
    def add_nodes(self, nqubit: int, data: Data) -> None:
        """
        Add nodes (qubits) to the state and initialize them in a specified state.

        Parameters
        ----------
        nqubit : int
            The number of qubits to add to the state.

        data : Data, optional
            The state in which to initialize the newly added nodes. The supported forms
            of state specification depend on the backend implementation.

        See :meth:`Backend.add_nodes` for further details.
        """

    @abstractmethod
    def entangle(self, edge: tuple[int, int]) -> None:
        """Connect graph nodes.

        Parameters
        ----------
        edge : tuple of int
            (control, target) qubit indices
        """

    @abstractmethod
    def evolve(self, op: Matrix, qargs: Sequence[int]) -> None:
        """Apply a multi-qubit operation.

        Parameters
        ----------
        op : numpy.ndarray
            2^n*2^n matrix
        qargs : list of int
            target qubits' indices
        """

    @abstractmethod
    def evolve_single(self, op: Matrix, i: int) -> None:
        """Apply a single-qubit operation.

        Parameters
        ----------
        op : numpy.ndarray
            2*2 matrix
        i : int
            qubit index
        """

    @abstractmethod
    def expectation_single(self, op: Matrix, loc: int) -> complex:
        """Return the expectation value of single-qubit operator.

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

    @abstractmethod
    def remove_qubit(self, qarg: int) -> None:
        """Remove a separable qubit from the system."""

    @abstractmethod
    def swap(self, qubits: tuple[int, int]) -> None:
        """Swap qubits.

        Parameters
        ----------
        qubits : tuple of int
            (control, target) qubit indices
        """


def _op_mat_from_result(
    vec: tuple[ExpressionOrFloat, ExpressionOrFloat, ExpressionOrFloat], result: Outcome, symbolic: bool = False
) -> Matrix:
    r"""Return the operator :math:`\tfrac{1}{2}(I + (-1)^r \vec{v}\cdot\vec{\sigma})`.

    Parameters
    ----------
    vec : tuple[float, float, float]
        Cartesian components of a unit vector.
    result : bool
        Measurement result ``r``.
    symbolic : bool, optional
        If ``True`` return an array of ``object`` dtype.

    Returns
    -------
    numpy.ndarray
        2x2 operator acting on the measured qubit.
    """
    sign = (-1) ** result
    if symbolic:
        op_mat_symbolic: npt.NDArray[np.object_] = np.eye(2, dtype=np.object_) / 2
        for i, t in enumerate(vec):
            op_mat_symbolic += sign * t * Clifford(i + 1).matrix / 2
        return op_mat_symbolic
    op_mat_complex: npt.NDArray[np.complex128] = np.eye(2, dtype=np.complex128) / 2
    x, y, z = vec
    # mypy requires each of x, y, and z to be tested explicitly for it to infer
    # that they are instances of `SupportsFloat`.
    # In particular, using a loop or comprehension like
    # `not all(isinstance(v, SupportsFloat) for v in (x, y, z))` is not supported.
    if not isinstance(x, SupportsFloat) or not isinstance(y, SupportsFloat) or not isinstance(z, SupportsFloat):
        raise TypeError("Vector of float expected with symbolic = False")
    float_vec = [x, y, z]
    for i, t in enumerate(float_vec):
        op_mat_complex += sign * t * Clifford(i + 1).matrix / 2
    return op_mat_complex


def perform_measure(
    qubit: int,
    plane: Plane,
    angle: ExpressionOrFloat,
    state: DenseState,
    rng: Generator,
    pr_calc: bool = True,
    symbolic: bool = False,
) -> Outcome:
    """Perform measurement of a qubit."""
    vec = plane.polar(angle)
    if pr_calc:
        op_mat = _op_mat_from_result(vec, 0, symbolic=symbolic)
        prob_0 = state.expectation_single(op_mat, qubit)
        result = outcome(rng.random() > abs(prob_0))
        if result:
            op_mat = _op_mat_from_result(vec, 1, symbolic=symbolic)
    else:
        # choose the measurement result randomly
        result = rng.choice([0, 1])
        op_mat = _op_mat_from_result(vec, result, symbolic=symbolic)
    state.evolve_single(op_mat, qubit)
    return result


StateT_co = TypeVar("StateT_co", bound="BackendState", covariant=True)


@dataclass(frozen=True)
class Backend(Generic[StateT_co]):
    """
    Abstract base class for all quantum backends.

    A backend is responsible for managing a quantum system, including the set of active
    qubits (nodes), their initialization, evolution, and measurement. It defines the
    interface through which high-level quantum programs interact with the underlying
    simulation or hardware model.

    Concrete subclasses implement specific state representations and simulation strategies,
    such as dense state vectors, density matrices, or tensor networks.

    Responsibilities of a backend typically include:
    - Managing a dynamic set of qubits (nodes) and their state
    - Applying quantum gates or operations
    - Performing measurements and returning classical outcomes
    - Tracking and exposing the underlying quantum state

    Examples of concrete subclasses include:
    - `StatevecBackend` (pure states via state vectors)
    - `DensityMatrixBackend` (mixed states via density matrices)
    - `TensorNetworkBackend` (compressed states via tensor networks)

    Parameters
    ----------
    state : BackendState
        internal state of the backend: instance of :class:`Statevec`, :class:`DensityMatrix`, or :class:`MBQCTensorNet`.

    Notes
    -----
    This class is abstract and should not be instantiated directly.

    The class hierarchy of states mirrors the class hierarchy of backends:
    - `FullStateBackend` and `TensorNetworkBackend` are subclasses of `Backend`,
      and `DenseState` and `MBQCTensorNet` are subclasses of `BackendState`.
    - `StatevecBackend` and `DensityMatrixBackend` are subclasses of `FullStateBackend`,
      and `Statevec` and `DensityMatrix` are subclasses of `DenseState`.

    The type variable `StateT_co` specifies the type of the ``state`` field, so that subclasses
    provide a precise type for this field:
    - `StatevecBackend` is a subtype of ``Backend[Statevec]``.
    - `DensityMatrixBackend` is a subtype of ``Backend[DensityMatrix]``.
    - `TensorNetworkBackend` is a subtype of ``Backend[MBQCTensorNet]``.

    The type variables `StateT_co` and `DenseStateT_co` are declared as covariant.
    That is, ``Backend[T1]`` is a subtype of ``Backend[T2]`` if ``T1`` is a subtype of ``T2``.
    This means that `StatevecBackend`, `DensityMatrixBackend`, and `TensorNetworkBackend` are
    all subtypes of ``Backend[BackendState]``.
    This covariance is sound because backends are frozen dataclasses; thus, the type of
    ``state`` cannot be changed after instantiation.

    See Also
    --------
    :class:`BackendState`, :`class:`FullStateBackend`, :class:`StatevecBackend`, :class:`DensityMatrixBackend`, :class:`TensorNetworkBackend`
    """

    # `init=False` is required because `state` cannot appear in a contravariant position
    # (specifically, as a parameter of `__init__`) since `StateT_co` is covariant.
    state: StateT_co = dataclasses.field(init=False)

    @abstractmethod
    def add_nodes(self, nodes: Sequence[int], data: Data = BasicStates.PLUS) -> None:
        r"""
        Add new nodes (qubits) to the backend and initialize them in a specified state.

        Parameters
        ----------
        nodes : Sequence[int]
            A list of node indices to add to the backend. These indices can be any
            integer values but must be fresh: each index must be distinct from all
            previously added nodes.

        data : Data, optional
            The state in which to initialize the newly added nodes. The supported forms
            of state specification depend on the backend implementation.

            All backends must support the basic predefined states in ``BasicStates``.

            - If a single basic state is provided, all new nodes are initialized in that state.
            - If a list of basic states is provided, it must match the length of ``nodes``, and
              each node is initialized with its corresponding state.

            Some backends support other forms of state specification.

            - ``StatevecBackend`` supports arbitrary state vectors:
                - A single-qubit state vector will be broadcast to all nodes.
                - A multi-qubit state vector of dimension :math:`2^n`, where :math:`n = \mathrm{len}(nodes)`,
                  initializes the new nodes jointly.

            - ``DensityMatrixBackend`` supports both state vectors and density matrices:
                - State vectors are handled as in ``StatevecBackend``, and converted to
                  density matrices.
                - A density matrix must have shape :math:`2^n \times 2^n`, where :math:`n = \mathrm{len}(nodes)`,
                  and is used to jointly initialize the new nodes.

        Notes
        -----
        Previously existing nodes remain unchanged.
        """

    def apply_channel(self, channel: KrausChannel, qargs: Collection[int]) -> None:
        """Apply channel to the state.

        Parameters
        ----------
            qargs : list of ints. Target qubits
        """
        _ = self  # silence PLC0105
        _ = channel  # silence ARG002
        _ = qargs
        raise NoiseNotSupportedError

    @abstractmethod
    def apply_clifford(self, node: int, clifford: Clifford) -> None:
        """Apply single-qubit Clifford gate, specified by vop index specified in graphix.clifford.CLIFFORD."""

    @abstractmethod
    def correct_byproduct(self, cmd: command.X | command.Z, measure_method: MeasureMethod) -> None:
        """Byproduct correction correct for the X or Z byproduct operators, by applying the X or Z gate."""

    @abstractmethod
    def entangle_nodes(self, edge: tuple[int, int]) -> None:
        """Apply CZ gate to two connected nodes.

        Parameters
        ----------
        edge : tuple (i, j)
            a pair of node indices
        """

    @abstractmethod
    def finalize(self, output_nodes: Iterable[int]) -> None:
        """To be run at the end of pattern simulation."""

    @abstractmethod
    def measure(self, node: int, measurement: Measurement) -> Outcome:
        """Perform measurement of a node and trace out the qubit.

        Parameters
        ----------
        node: int
        measurement: Measurement
        """


DenseStateT_co = TypeVar("DenseStateT_co", bound="DenseState", covariant=True)


@dataclass(frozen=True)
class FullStateBackend(Backend[DenseStateT_co], Generic[DenseStateT_co]):
    """
    Abstract base class for backends that represent quantum states explicitly in memory.

    This class defines common functionality for backends that store the entire quantum
    state as a dense array—either as a state vector (pure state) or a density matrix
    (mixed state)—and perform quantum operations using standard linear algebra. It is
    designed to be the shared base class of `StatevecBackend` and `DensityMatrixBackend`.

    In contrast to :class:`TensorNetworkBackend`, which uses structured and compressed
    representations (e.g., matrix product states) to scale to larger systems,
    `FullStateBackend` subclasses simulate quantum systems by maintaining the full
    state in memory. This approach enables straightforward implementation of gates,
    measurements, and noise models, but scales exponentially with the number of qubits.

    This class is not meant to be instantiated directly.

    Parameters
    ----------
    node_index : NodeIndex, optional
        Mapping between node numbers and qubit indices in the internal state of the backend.
    pr_calc : bool, optional
        Whether or not to compute the probability distribution before choosing the measurement outcome.
        If False, measurements yield results 0/1 with 50% probabilities each.
    rng : Generator, optional
        Random number generator used to sample measurement outcomes.
    symbolic : bool, optional
        If True, support arbitrary objects (typically, symbolic expressions) in matrices.

    See Also
    --------
    :class:`StatevecBackend`, :class:`DensityMatrixBackend`, :class:`TensorNetworkBackend`
    """

    node_index: NodeIndex = dataclasses.field(default_factory=NodeIndex)
    pr_calc: bool = True
    rng: Generator = dataclasses.field(default_factory=ensure_rng)
    symbolic: bool = False

    @override
    def add_nodes(self, nodes: Sequence[int], data: Data = BasicStates.PLUS) -> None:
        """
        Add new nodes (qubits) to the backend and initialize them in a specified state.

        Parameters
        ----------
        nodes : Sequence[int]
            A list of node indices to add to the backend. These indices can be any
            integer values but must be fresh: each index must be distinct from all
            previously added nodes.

        data : Data, optional
            The state in which to initialize the newly added nodes. The supported forms
            of state specification depend on the backend implementation.

        See :meth:`Backend.add_nodes` for further details.
        """
        self.state.add_nodes(nqubit=len(nodes), data=data)
        self.node_index.extend(nodes)

    @override
    def entangle_nodes(self, edge: tuple[int, int]) -> None:
        """Apply CZ gate to two connected nodes.

        Parameters
        ----------
        edge : tuple (i, j)
            a pair of node indices
        """
        target = self.node_index.index(edge[0])
        control = self.node_index.index(edge[1])
        self.state.entangle((target, control))

    @override
    def measure(self, node: int, measurement: Measurement) -> Outcome:
        """Perform measurement of a node and trace out the qubit.

        Parameters
        ----------
        node: int
        measurement: Measurement
        """
        loc = self.node_index.index(node)
        result = perform_measure(
            loc,
            measurement.plane,
            measurement.angle,
            self.state,
            rng=self.rng,
            pr_calc=self.pr_calc,
            symbolic=self.symbolic,
        )
        self.node_index.remove(node)
        self.state.remove_qubit(loc)
        return result

    @override
    def correct_byproduct(self, cmd: command.X | command.Z, measure_method: MeasureMethod) -> None:
        """Byproduct correction correct for the X or Z byproduct operators, by applying the X or Z gate."""
        if np.mod(sum(measure_method.get_measure_result(j) for j in cmd.domain), 2) == 1:
            op = Ops.X if cmd.kind == CommandKind.X else Ops.Z
            self.apply_single(node=cmd.node, op=op)

    @override
    def apply_channel(self, channel: KrausChannel, qargs: Collection[int]) -> None:
        """Apply channel to the state.

        Parameters
        ----------
            qargs : list of ints. Target qubits
        """
        indices = [self.node_index.index(i) for i in qargs]
        self.state.apply_channel(channel, indices)

    def apply_single(self, node: int, op: Matrix) -> None:
        """Apply a single gate to the state."""
        index = self.node_index.index(node)
        self.state.evolve_single(op=op, i=index)

    @override
    def apply_clifford(self, node: int, clifford: Clifford) -> None:
        """Apply single-qubit Clifford gate, specified by vop index specified in graphix.clifford.CLIFFORD."""
        loc = self.node_index.index(node)
        self.state.evolve_single(clifford.matrix, loc)

    def sort_qubits(self, output_nodes: Iterable[int]) -> None:
        """Sort the qubit order in internal statevector."""
        for i, ind in enumerate(output_nodes):
            if self.node_index.index(ind) != i:
                move_from = self.node_index.index(ind)
                self.state.swap((i, move_from))
                self.node_index.swap(i, move_from)

    @override
    def finalize(self, output_nodes: Iterable[int]) -> None:
        """To be run at the end of pattern simulation."""
        self.sort_qubits(output_nodes)

    @property
    def nqubit(self) -> int:
        """Return the number of qubits of the current state."""
        return self.state.nqubit
