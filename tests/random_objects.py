import numpy as np
import numpy.typing as npt
import scipy.linalg
from graphix.kraus import Channel


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
# TODO implement and checks


def rand_gauss_cpx_mat(dim: int, sig: float = 1 / np.sqrt(2)) -> npt.NDArray:

    # [Mis12] Miszczak, Generating and using truly random quantum states in Mathematica, Computer Physics Communications 183 1, 118-124 (2012). doi:10.1016/j.cpc.2011.08.002.

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

    rank : int
        Choi rank ie the number of Kraus operators.
        WARNING Qutip mentions complete positivity issues for not full rank

    sig : cf rand_cpx


    norm : float

    """

    # TODO check complete positivity!
    # use Cholesky decomp?
    # https://math.stackexchange.com/questions/87528/a-practical-way-to-check-if-a-matrix-is-positive-definite
    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.cholesky.html

    if rank is None:
        rank = dim**2

    if sig == "ginibre":
        sig = 1.0 / np.sqrt(2 * dim)

    assert 1 <= rank <= dim**2

    pre_kraus_list = [rand_gauss_cpx_mat(dim=dim, sig=sig) for _ in range(rank)]
    Hmat = np.sum([m.transpose().conjugate() @ m for m in pre_kraus_list], axis=0)
    kraus_list = np.array(pre_kraus_list) @ scipy.linalg.inv(scipy.linalg.sqrtm(Hmat))

    return Channel([{"parameter": 1.0, "operator": kraus_list[i]} for i in range(rank)])
