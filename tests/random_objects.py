import numpy as np
import numpy.typing as npt
import scipy.linalg

from graphix.channels import Channel
from graphix.sim.density_matrix import DensityMatrix


def rand_herm(l: int):
    """
    generate random hermitian matrix of size l*l
    """
    tmp = np.random.rand(l, l) + 1j * np.random.rand(l, l)
    return tmp + tmp.conj().T


def rand_unit(l: int):
    """
    generate random unitary matrix of size l*l from hermitian matrix
    """
    return scipy.linalg.expm(1j * rand_herm(l))


UNITS = np.array([1, 1j])


def rand_dm(dim: int, rank: int = None, dm_dtype=True) -> DensityMatrix:
    """Returns a "density matrix" as a DensityMatrix object ie a positive-semidefinite (hence Hermitian) matrix with unit trace
    Note, not a proper DM since its dim can be something else than a power of 2.
    The rank is random between 1 (pure) and dim if not specified
    Thanks to Ulysse Chabaud.

    :param dim: Linear dimension of the (square) matrix
    :type dim: int
    :param rank: If rank not specified then random between 1 and matrix dimension
        If rank is one : then pure state else mixed state. Defaults to None
    :type rank: int, optional
    :param dm_dtype: If True returns a :class:`graphix.sim.density_matrix.DensityMatrix` or a numpy.ndarray if False. Defaults to True.
    :type dm_dtype: bool, optional
    :return: Random density matrix as a :class:`graphix.sim.density_matrix.DensityMatrix` object or a numpy.ndarray.
    :rtype: :class:`graphix.sim.density_matrix.DensityMatrix` or numpy.ndarray.

    """

    # if not provided, use a random value.
    if rank is None:
        rank = np.random.randint(1, dim + 1)

    evals = np.random.rand(rank)

    padded_evals = np.zeros(dim)
    padded_evals[: len(evals)] = evals

    dm = np.diag(padded_evals / np.sum(padded_evals))

    randU = rand_unit(dim)
    dm = randU @ dm @ randU.transpose().conj()

    if dm_dtype:
        # will raise an error if incorrect dimension
        return DensityMatrix(data=dm)
    else:
        return dm


def rand_gauss_cpx_mat(dim: int, sig: float = 1 / np.sqrt(2)) -> npt.NDArray:

    """
    Returns a square array of standard normal complex random variates.
    Code from QuTiP: https://qutip.org/docs/4.0.2/modules/qutip/random_objects.html

    Parameters
    ----------
    dim : int
        Linear dimension of the (square) matrix
    sig : float
        standard deviation of random variates.
        ``sig = 'ginibre`` draws from the Ginibre ensemble ie  sig = 1 / sqrt(2 * dim).

    """

    if sig == "ginibre":
        sig = 1.0 / np.sqrt(2 * dim)

    return np.sum(np.random.normal(loc=0.0, scale=sig, size=((dim,) * 2 + (2,))) * UNITS, axis=-1)


def rand_channel_kraus(dim: int, rank: int = None, sig: float = 1 / np.sqrt(2)) -> Channel:

    """
    Returns a random :class:`graphix.sim.channels.Channel`object of given dimension and rank following the method of
    [KNPPZ21] Kukulski, Nechita, Pawela, Puchała, Życzkowsk https://arxiv.org/pdf/2011.02994.pdf

    Parameters
    ----------
    dim : int
        Linear dimension of the (square) matrix of each Kraus operator.
        Only square operators so far.

    rank : int (default to full rank dimension**2)
        Choi rank ie the number of Kraus operators. Must be between one and dim**2.

    sig : see rand_cpx

    """

    if rank is None:
        rank = dim**2

    if sig == "ginibre":
        sig = 1.0 / np.sqrt(2 * dim)

    if not isinstance(rank, int):
        raise TypeError("The rank of a Kraus expansion must be an integer.")

    if not 1 <= rank:
        raise ValueError("The rank of a Kraus expansion must be greater than 1.")

    pre_kraus_list = [rand_gauss_cpx_mat(dim=dim, sig=sig) for _ in range(rank)]
    Hmat = np.sum([m.transpose().conjugate() @ m for m in pre_kraus_list], axis=0)
    kraus_list = np.array(pre_kraus_list) @ scipy.linalg.inv(scipy.linalg.sqrtm(Hmat))

    return Channel([{"parameter": 1.0 + 0.0 * 1j, "operator": kraus_list[i]} for i in range(rank)])
