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


# code from Qutip
# https://qutip.org/docs/4.0.2/modules/qutip/random_objects.html

UNITS = np.array([1, 1j])
# TODO implement random DM distinct from statevector? allows for non-pure states?
# not necessary dm since arbitrary dim here.


def rand_dm(dim: int, rank: int = None):
    """
    Returns a "density matrix" ie a positive-semidefinite (hence Hermitian) matrix with unit trace
    Note, not a proper DM since its dim can be something else than a power of 2.
    The rank is random between 1 (pure) and dim if not specified
    Thanks to Ulysse Chabaud.

    Parameters
    ----------
    dim : int
        Linear dimension of the (square) matrix
    rank : int (default None)
        If rank not specified then random between 1 and matrix dimension
        If rank is one : then pure state else mixed state.
    """
    # if not provided, use a random value.
    if rank is None:
        rank = np.random.randint(1, dim + 1)

    # randomly choose eigenvalues rank eigenvalues between 0 and 1 (excluded)
    evals = np.random.rand(rank)

    # pad with zeros everywhere else to match dimensions
    padded_evals = np.zeros(dim)
    padded_evals[: len(evals)] = evals

    # normalize by the trace
    # np.diag: if arg is 1D array, constructs the 2D array with the corresponding diagonal
    # if arg is 2D, extracts the diagonal

    dm = np.diag(padded_evals / np.sum(padded_evals))

    # generate a random unitary

    randU = rand_unit(dim)
    dm = randU @ dm @ randU.transpose().conj()

    # or just [Miszczak11] but no control over the rank.
    # tmp = np.random.rand(l, l) + 1j * np.random.rand(l, l)
    # np.dot(tmp, tmp.conj().T)

    return DensityMatrix(data=dm)


def rand_gauss_cpx_mat(dim: int, sig: float = 1 / np.sqrt(2)) -> npt.NDArray:

    # [Mis12] Miszczak, Generating and using truly random quantum states in Mathematica, Computer Physics Communications 183 1, 118-124 (2012). doi:10.1016/j.cpc.2011.08.002.
    # Majumdar Scher https://arxiv.org/abs/1904.01813 for Ginibre ensemble definition
    """
    Returns an array of standard normal complex random variates.
    The Ginibre ensemble corresponds to setting ``norm = 1`` [Mis12]_.

    Parameters
    ----------
    dim : int
        Linear dimension of the (square) matrix
    sig : float
        standard deviation of random variates, or 'ginibre' to draw
        from the Ginibre ensemble.
        1/ sqrt(2) means each entry as variance 1. (Cpx vriance is sum of re and im variance)
        Formally complex Ginibre ensemble is Re and Im have 1/(2N) variance [LGCCKMS19] https://arxiv.org/pdf/1904.01813.pdf
        Complex variance : is sum of variances. Currently Ginibre.
        [Mis12] has normal variance of 1 so sum of both makes variance 2
    """
    # NOTE currently only allow for square matrices. See Qutip for shape.
    if sig == "ginibre":
        sig = 1.0 / np.sqrt(2 * dim)

    return np.sum(np.random.normal(loc=0.0, scale=sig, size=((dim,) * 2 + (2,))) * UNITS, axis=-1)


def rand_channel_kraus(dim: int, rank: int = None, sig: float = 1 / np.sqrt(2)) -> Channel:

    # [KNPPZ21] Kukulski, Nechita, Pawela, Puchała, Życzkowsk https://arxiv.org/pdf/2011.02994.pdf

    """
    Returns

    Parameters
    ----------
     dim : int
        Linear dimension of the (square) matrix of each Kraus operator.
        Only square operators (so far)!

    rank : int (default dimension**2 (maximum rank))
        Choi rank ie the number of Kraus operators. Must be between one and dim**2.
        WARNING Qutip mentions complete positivity issues for not full rank

    sig : cf rand_cpx


    norm : float

    """

    # TODO check complete positivity?
    # immediate from Kraus decomposition
    # use Cholesky decomp?
    # https://math.stackexchange.com/questions/87528/a-practical-way-to-check-if-a-matrix-is-positive-definite
    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.cholesky.html

    # default is full rank.
    if rank is None:
        rank = dim**2

    if sig == "ginibre":
        sig = 1.0 / np.sqrt(2 * dim)

    # Condition in def 2 and eq 15 of [KNPPZ21]
    # the smaller than d**2 checked in the Channel class __init__

    if not isinstance(rank, int):
        raise TypeError("The rank of a Kraus expansion must be an integer.")

    if not 1 <= rank:
        raise ValueError("The rank of a Kraus expansion must be greater than 1.")

    pre_kraus_list = [rand_gauss_cpx_mat(dim=dim, sig=sig) for _ in range(rank)]
    Hmat = np.sum([m.transpose().conjugate() @ m for m in pre_kraus_list], axis=0)
    kraus_list = np.array(pre_kraus_list) @ scipy.linalg.inv(scipy.linalg.sqrtm(Hmat))

    return Channel([{"parameter": 1.0 + 0.0 * 1j, "operator": kraus_list[i]} for i in range(rank)])
