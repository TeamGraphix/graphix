"""Functions to generate various random objects."""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import scipy.linalg
from scipy.stats import unitary_group

from graphix.channels import KrausChannel, KrausData
from graphix.ops import Ops
from graphix.rng import ensure_rng
from graphix.transpiler import Circuit

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from typing import Literal, TypeAlias

    from numpy.random import Generator

    from graphix.parameter import Parameter

    IntLike: TypeAlias = int | np.integer


def rand_herm(sz: IntLike, rng: Generator | None = None) -> npt.NDArray[np.complex128]:
    """Generate random hermitian matrix of size sz*sz."""
    rng = ensure_rng(rng)
    tmp = rng.random(size=(sz, sz)) + 1j * rng.random(size=(sz, sz))
    return tmp + tmp.conj().T


def rand_unit(sz: IntLike, rng: Generator | None = None) -> npt.NDArray[np.complex128]:
    """Generate haar random unitary matrix of size sz*sz."""
    rng = ensure_rng(rng)
    if sz == 1:
        return np.array([np.exp(1j * rng.random(size=1) * 2 * np.pi)])
    return unitary_group.rvs(int(sz), random_state=rng)


UNITS = np.array([1, 1j])


def rand_dm(dim: IntLike, rng: Generator | None = None, rank: IntLike | None = None) -> npt.NDArray[np.complex128]:
    """Generate random density matrices (positive semi-definite matrices with unit trace).

    Returns either a :class:`graphix.sim.density_matrix.DensityMatrix` or a :class:`np.ndarray` depending on the parameter *dm_dtype*.

    :param dim: Linear dimension of the (square) matrix
    :type dim: int
    :param rank: Rank of the density matrix (1 = pure state). If not specified then sent to dim (maximal rank).
        Defaults to None
    :type rank: int, optional
    :return: the density matrix in the specified format.
    :rtype: DensityMatrix | np.ndarray

    .. note::
        Thanks to Ulysse Chabaud.
    """
    rng = ensure_rng(rng)

    if rank is None:
        rank = dim

    evals = rng.random(size=rank)

    padded_evals = np.zeros(dim)
    padded_evals[: len(evals)] = evals

    dm = np.diag(padded_evals / np.sum(padded_evals))

    rand_u = rand_unit(dim)
    return rand_u @ dm @ rand_u.transpose().conj()


if TYPE_CHECKING:
    _SIG: TypeAlias = float | Literal["ginibre"] | None


def _make_sig(sig: _SIG, dim: IntLike) -> float:
    if sig is None:
        # B008 Do not perform function call in argument defaults
        return 1 / math.sqrt(2)
    if sig == "ginibre":
        return 1 / math.sqrt(2 * dim)
    return sig


def rand_gauss_cpx_mat(dim: IntLike, rng: Generator | None = None, sig: _SIG = None) -> npt.NDArray[np.complex128]:
    """Return a square array of standard normal complex random variates.

    Code from QuTiP: https://qutip.org/docs/4.0.2/modules/qutip/random_objects.html

    Parameters
    ----------
    dim : int
        Linear dimension of the (square) matrix
    sig : float | Literal["ginibre"] | None
        standard deviation of random variates.
        ``sig = 'ginibre`` draws from the Ginibre ensemble ie  sig = 1 / sqrt(2 * dim).

    """
    rng = ensure_rng(rng)

    result: npt.NDArray[np.complex128] = np.sum(
        rng.normal(loc=0.0, scale=_make_sig(sig, dim), size=((dim,) * 2 + (2,))) * UNITS, axis=-1
    )
    return result


def rand_channel_kraus(
    dim: int, rng: Generator | None = None, rank: int | None = None, sig: _SIG = None
) -> KrausChannel:
    """Return a random :class:`graphix.sim.channels.KrausChannel` object of given dimension and rank.

    Following the method of
    [KNPPZ21] Kukulski, Nechita, Pawela, Puchała, Życzkowsk https://arxiv.org/pdf/2011.02994.pdf

    Parameters
    ----------
    dim : int
        Linear dimension of the (square) matrix of each Kraus operator.
        Only square operators so far.

    rank : int (default to full *rank dimension**2*)
        Choi rank ie the number of Kraus operators. Must be between one and *dim**2*.

    sig : see rand_cpx

    """
    rng = ensure_rng(rng)

    if rank is None:
        rank = dim**2

    if not rank >= 1:
        raise ValueError("The rank of a Kraus expansion must be greater or equal than 1.")

    pre_kraus_list = [rand_gauss_cpx_mat(dim=dim, sig=_make_sig(sig, dim)) for _ in range(rank)]
    h_mat = np.sum([m.transpose().conjugate() @ m for m in pre_kraus_list], axis=0)
    kraus_list = np.array(pre_kraus_list) @ scipy.linalg.inv(scipy.linalg.sqrtm(h_mat))

    return KrausChannel([KrausData(1.0 + 0.0 * 1j, kraus_list[i]) for i in range(rank)])


# or merge with previous with a "pauli" kwarg?
# continue here
def rand_pauli_channel_kraus(dim: int, rng: Generator | None = None, rank: int | None = None) -> KrausChannel:
    """Return a random Kraus channel operator."""
    rng = ensure_rng(rng)

    if not dim & (dim - 1) == 0:
        raise ValueError(f"The dimension must be a power of 2 and not {dim}.")

    nqb = int(np.log2(dim))

    # max number of ops (Choi rank) is d**2
    # default is full rank.
    if rank is None:
        rank = dim**2
    elif not rank >= 1:
        raise ValueError("The rank of a Kraus expansion must be an integer greater or equal than 1.")

    # full probability has to have dim**2 operators.
    prob_list = np.zeros(dim**2)
    # generate rank random numbers and normalize
    tmp_list = rng.uniform(size=rank)
    tmp_list /= tmp_list.sum()

    # fill the list and shuffle
    prob_list[:rank] = tmp_list
    rng.shuffle(prob_list)

    tensor_pauli_ops = Ops.build_tensor_pauli_ops(nqb)
    target_indices = np.nonzero(prob_list)

    params = prob_list[target_indices]
    ops = tensor_pauli_ops[target_indices]

    # TODO see how to use zip and dict to convert from tuple to dict
    # https://www.tutorialspoint.com/How-I-can-convert-a-Python-Tuple-into-Dictionary

    data = [KrausData(np.sqrt(params[i]), ops[i]) for i in range(rank)]

    # NOTE retain a strong probability on the identity or not?
    # think we don't really care

    return KrausChannel(data)


def _first_rotation(circuit: Circuit, nqubits: int, rng: Generator) -> None:
    """Apply an initial random :math:`R_X` rotation to each qubit.

    Parameters
    ----------
    circuit : Circuit
        Circuit to modify in place.
    nqubits : int
        Number of qubits.
    rng : numpy.random.Generator
        Random number generator used to sample rotation angles.
    """
    for qubit in range(nqubits):
        circuit.rx(qubit, rng.random())


def _mid_rotation(circuit: Circuit, nqubits: int, rng: Generator) -> None:
    """Apply a random :math:`R_X` then :math:`R_Z` rotation to each qubit.

    Parameters
    ----------
    circuit : Circuit
        Circuit to modify in place.
    nqubits : int
        Number of qubits.
    rng : numpy.random.Generator
        Random number generator used to sample rotation angles.
    """
    for qubit in range(nqubits):
        circuit.rx(qubit, rng.random())
        circuit.rz(qubit, rng.random())


def _last_rotation(circuit: Circuit, nqubits: int, rng: Generator) -> None:
    """Apply a final random :math:`R_Z` rotation to each qubit.

    Parameters
    ----------
    circuit : Circuit
        Circuit to modify in place.
    nqubits : int
        Number of qubits.
    rng : numpy.random.Generator
        Random number generator used to sample rotation angles.
    """
    for qubit in range(nqubits):
        circuit.rz(qubit, rng.random())


def _entangler(circuit: Circuit, pairs: Iterable[tuple[int, int]]) -> None:
    """Apply CNOT gates between qubit pairs.

    Parameters
    ----------
    circuit : Circuit
        Circuit to modify in place.
    pairs : Iterable[tuple[int, int]]
        Pairs of control and target qubits for CNOT operations.
    """
    for a, b in pairs:
        circuit.cnot(a, b)


def _entangler_rzz(circuit: Circuit, pairs: Iterable[tuple[int, int]], rng: Generator) -> None:
    """Apply random :math:`R_{ZZ}` gates between qubit pairs.

    Parameters
    ----------
    circuit : Circuit
        Circuit to modify in place.
    pairs : Iterable[tuple[int, int]]
        Pairs of qubits on which to apply the :math:`R_{ZZ}` gate.
    rng : numpy.random.Generator
        Random number generator used to sample rotation angles.
    """
    for a, b in pairs:
        circuit.rzz(a, b, rng.random())


def rand_gate(
    nqubits: int,
    depth: int,
    pairs: Iterable[tuple[int, int]],
    rng: Generator | None = None,
    *,
    use_rzz: bool = False,
) -> Circuit:
    """Return a random gate composed of single-qubit rotations and entangling operations.

    Parameters
    ----------
    nqubits : int
        Number of qubits in the circuit.
    depth : int
        Depth of alternating rotation and entangling layers.
    pairs : Iterable[tuple[int, int]]
        Pairs of qubits used for entangling operations.
    rng : numpy.random.Generator, optional
        Random number generator used to sample rotation angles. If ``None``, a
        default generator is created.
    use_rzz : bool, optional
        If ``True`` use :math:`R_{ZZ}` gates as entanglers instead of CNOT.

    Returns
    -------
    Circuit
        The generated random circuit.
    """
    rng = ensure_rng(rng)
    circuit = Circuit(nqubits)
    _first_rotation(circuit, nqubits, rng)
    _entangler(circuit, pairs)
    for _ in range(depth - 1):
        _mid_rotation(circuit, nqubits, rng)
        if use_rzz:
            _entangler_rzz(circuit, pairs, rng)
        else:
            _entangler(circuit, pairs)
    _last_rotation(circuit, nqubits, rng)
    return circuit


def _genpair(n_qubits: int, count: int, rng: Generator) -> Iterator[tuple[int, int]]:
    """Yield random pairs of qubit indices.

    Parameters
    ----------
    n_qubits : int
        Number of available qubits.
    count : int
        Number of pairs to generate.
    rng : numpy.random.Generator
        Random number generator for selecting qubits.

    Yields
    ------
    tuple[int, int]
        Randomly selected qubit pair.
    """
    choice = list(range(n_qubits))
    for _ in range(count):
        rng.shuffle(choice)
        x, y = choice[:2]
        yield (x, y)


def _gentriplet(n_qubits: int, count: int, rng: Generator) -> Iterator[tuple[int, int, int]]:
    """Yield random triplets of qubit indices.

    Parameters
    ----------
    n_qubits : int
        Number of available qubits.
    count : int
        Number of triplets to generate.
    rng : numpy.random.Generator
        Random number generator for selecting qubits.

    Yields
    ------
    tuple[int, int, int]
        Randomly selected qubit triplet.
    """
    choice = list(range(n_qubits))
    for _ in range(count):
        rng.shuffle(choice)
        x, y, z = choice[:3]
        yield (x, y, z)


def rand_circuit(
    nqubits: int,
    depth: int,
    rng: Generator | None = None,
    *,
    use_cz: bool = True,
    use_rzz: bool = False,
    use_ccx: bool = False,
    parameters: Iterable[Parameter] | None = None,
) -> Circuit:
    """Return a random parameterized circuit used for testing or benchmarking.

    Parameters
    ----------
    nqubits : int
        Number of qubits in the circuit.
    depth : int
        Number of alternating entangling and single-qubit layers.
    rng : numpy.random.Generator, optional
        Random number generator. A default generator is created if ``None``.
    use_cz : bool, optional
        If ``True`` add CZ gates in each layer (default: ``True``).
    use_rzz : bool, optional
        If ``True`` add :math:`R_{ZZ}` gates in each layer (default: ``False``).
    use_ccx : bool, optional
        If ``True`` add CCX gates in each layer (default: ``False``).
    parameters : Iterable[Parameter], optional
        Parameters used for randomly chosen rotation gates.

    Returns
    -------
    Circuit
        The generated random circuit.
    """
    rng = ensure_rng(rng)
    circuit = Circuit(nqubits)
    parametric_gate_choice = (
        functools.partial(rotation, angle=parameter)
        for rotation in (circuit.rx, circuit.ry, circuit.rz)
        for parameter in parameters or []
    )
    gate_choice: tuple[Callable[[int], None], ...] = (
        functools.partial(circuit.ry, angle=np.pi / 4),
        functools.partial(circuit.rz, angle=-np.pi / 4),
        functools.partial(circuit.rx, angle=-np.pi / 4),
        circuit.h,
        circuit.s,
        circuit.x,
        circuit.z,
        circuit.y,
        *parametric_gate_choice,
    )
    for _ in range(depth):
        for j, k in _genpair(nqubits, 2, rng):
            circuit.cnot(j, k)
        if use_cz:
            for j, k in _genpair(nqubits, 2, rng):
                circuit.cz(j, k)
        if use_rzz:
            for j, k in _genpair(nqubits, 2, rng):
                circuit.rzz(j, k, np.pi / 4)
        if use_ccx:
            for j, k, l in _gentriplet(nqubits, 2, rng):
                circuit.ccx(j, k, l)
        for j, k in _genpair(nqubits, 4, rng):
            circuit.swap(j, k)
        for j in range(nqubits):
            ind = rng.integers(len(gate_choice))
            gate_choice[ind](j)
    return circuit


def rand_state_vector(nqubits: int, rng: Generator | None = None) -> npt.NDArray[np.complex128]:
    """
    Generate a random normalized complex state vector of size 2^n.

    Parameters
    ----------
    nqubits : int
        The power of 2 for the vector size

    Returns
    -------
    numpy.ndarray
        Normalized complex vector of size 2^nqubits
    """
    rng = ensure_rng(rng)
    dim = 1 << nqubits  # 2**nqubits is typed Any
    real, imag = rng.random((2, dim)) - 0.5
    vec: npt.NDArray[np.complex128] = (real + 1j * imag).astype(np.complex128)
    return vec / np.linalg.norm(vec)
