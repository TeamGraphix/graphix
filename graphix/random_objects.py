"""Functions to generate various random objects."""

from __future__ import annotations

import functools
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
    from collections.abc import Iterable, Iterator

    from numpy.random import Generator

    from graphix.parameter import Parameter
    from graphix.sim.density_matrix import DensityMatrix


def rand_herm(sz: int, rng: Generator | None = None) -> npt.NDArray:
    """Generate random hermitian matrix of size sz*sz."""
    rng = ensure_rng(rng)
    tmp = rng.random(size=(sz, sz)) + 1j * rng.random(size=(sz, sz))
    return tmp + tmp.conj().T


def rand_unit(sz: int, rng: Generator | None = None) -> npt.NDArray:
    """Generate haar random unitary matrix of size sz*sz."""
    rng = ensure_rng(rng)
    if sz == 1:
        return np.array([np.exp(1j * rng.random(size=1) * 2 * np.pi)])
    return unitary_group.rvs(sz, random_state=rng)


UNITS = np.array([1, 1j])


def rand_dm(
    dim: int, rng: Generator | None = None, rank: int | None = None, dm_dtype=True
) -> DensityMatrix | npt.NDArray:
    """Generate random density matrices (positive semi-definite matrices with unit trace).

    Returns either a :class:`graphix.sim.density_matrix.DensityMatrix` or a :class:`np.ndarray` depending on the parameter `dm_dtype`.

    :param dim: Linear dimension of the (square) matrix
    :type dim: int
    :param rank: Rank of the density matrix (1 = pure state). If not specified then sent to dim (maximal rank).
        Defaults to None
    :type rank: int, optional
    :param dm_dtype: If `True` returns a :class:`graphix.sim.density_matrix.DensityMatrix` object. If `False`returns a :class:`np.ndarray`
    :type dm_dtype: bool, optional
    :return: the density matrix in the specified format.
    :rtype: DensityMatrix | np.ndarray

    .. note::
        Thanks to Ulysse Chabaud.
    .. warning::
        Note that setting `dm_dtype=False` allows to generate "density matrices" inconsistent with qubits i.e. with dimensions not being powers of 2.
    """
    rng = ensure_rng(rng)

    if rank is None:
        rank = dim

    evals = rng.random(size=rank)

    padded_evals = np.zeros(dim)
    padded_evals[: len(evals)] = evals

    dm = np.diag(padded_evals / np.sum(padded_evals))

    rand_u = rand_unit(dim)
    dm = rand_u @ dm @ rand_u.transpose().conj()

    if dm_dtype:
        from graphix.sim.density_matrix import DensityMatrix  # circumvent circular import

        # will raise an error if incorrect dimension
        return DensityMatrix(data=dm)
    return dm


def rand_gauss_cpx_mat(dim: int, rng: Generator | None = None, sig: float = 1 / np.sqrt(2)) -> npt.NDArray:
    """Return a square array of standard normal complex random variates.

    Code from QuTiP: https://qutip.org/docs/4.0.2/modules/qutip/random_objects.html

    Parameters
    ----------
    dim : int
        Linear dimension of the (square) matrix
    sig : float
        standard deviation of random variates.
        ``sig = 'ginibre`` draws from the Ginibre ensemble ie  sig = 1 / sqrt(2 * dim).

    """
    rng = ensure_rng(rng)

    if sig == "ginibre":
        sig = 1.0 / np.sqrt(2 * dim)

    return np.sum(rng.normal(loc=0.0, scale=sig, size=((dim,) * 2 + (2,))) * UNITS, axis=-1)


def rand_channel_kraus(
    dim: int, rng: Generator | None = None, rank: int | None = None, sig: float = 1 / np.sqrt(2)
) -> KrausChannel:
    """Return a random :class:`graphix.sim.channels.KrausChannel` object of given dimension and rank.

    Following the method of
    [KNPPZ21] Kukulski, Nechita, Pawela, Puchała, Życzkowsk https://arxiv.org/pdf/2011.02994.pdf

    Parameters
    ----------
    dim : int
        Linear dimension of the (square) matrix of each Kraus operator.
        Only square operators so far.

    rank : int (default to full `rank dimension**2`)
        Choi rank ie the number of Kraus operators. Must be between one and `dim**2`.

    sig : see rand_cpx

    """
    rng = ensure_rng(rng)

    if rank is None:
        rank = dim**2

    if sig == "ginibre":
        sig = 1.0 / np.sqrt(2 * dim)

    if not isinstance(rank, int):
        raise TypeError("The rank of a Kraus expansion must be an integer.")

    if not rank >= 1:
        raise ValueError("The rank of a Kraus expansion must be greater or equal than 1.")

    pre_kraus_list = [rand_gauss_cpx_mat(dim=dim, sig=sig) for _ in range(rank)]
    h_mat = np.sum([m.transpose().conjugate() @ m for m in pre_kraus_list], axis=0)
    kraus_list = np.array(pre_kraus_list) @ scipy.linalg.inv(scipy.linalg.sqrtm(h_mat))

    return KrausChannel([KrausData(1.0 + 0.0 * 1j, kraus_list[i]) for i in range(rank)])


# or merge with previous with a "pauli" kwarg?
### continue here
def rand_pauli_channel_kraus(dim: int, rng: Generator | None = None, rank: int | None = None) -> KrausChannel:
    """Return a random Kraus channel operator."""
    rng = ensure_rng(rng)

    if not isinstance(dim, int):
        raise ValueError(f"The dimension must be an integer and not {dim}.")

    if not dim & (dim - 1) == 0:
        raise ValueError(f"The dimension must be a power of 2 and not {dim}.")

    nqb = int(np.log2(dim))

    # max number of ops (Choi rank) is d**2
    # default is full rank.
    if rank is None:
        rank = dim**2
    else:
        if not isinstance(rank, int):
            raise TypeError("The rank of a Kraus expansion must be an integer.")
        if not rank >= 1:
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
    for qubit in range(nqubits):
        circuit.rx(qubit, rng.random())


def _mid_rotation(circuit: Circuit, nqubits: int, rng: Generator) -> None:
    for qubit in range(nqubits):
        circuit.rx(qubit, rng.random())
        circuit.rz(qubit, rng.random())


def _last_rotation(circuit: Circuit, nqubits: int, rng: Generator) -> None:
    for qubit in range(nqubits):
        circuit.rz(qubit, rng.random())


def _entangler(circuit: Circuit, pairs: Iterable[tuple[int, int]]) -> None:
    for a, b in pairs:
        circuit.cnot(a, b)


def _entangler_rzz(circuit: Circuit, pairs: Iterable[tuple[int, int]], rng: Generator) -> None:
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
    """Return a random gate."""
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
    choice = list(range(n_qubits))
    for _ in range(count):
        rng.shuffle(choice)
        x, y = choice[:2]
        yield (x, y)


def _gentriplet(n_qubits: int, count: int, rng: Generator) -> Iterator[tuple[int, int, int]]:
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
    use_rzz: bool = False,
    use_ccx: bool = False,
    parameters: Iterable[Parameter] | None = None,
) -> Circuit:
    """Return a random circuit."""
    rng = ensure_rng(rng)
    circuit = Circuit(nqubits)
    parametric_gate_choice = (
        functools.partial(rotation, angle=parameter)
        for rotation in (circuit.rx, circuit.ry, circuit.rz)
        for parameter in parameters or []
    )
    gate_choice = (
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
