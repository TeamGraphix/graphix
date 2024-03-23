import numpy as np

from graphix.linalg_validations import check_data_dims, check_data_normalization, check_data_values_type, check_rank
from graphix.ops import Ops


class KrausChannel:
    """quantum channel class in the Kraus representation.
    Defined by Kraus operators :math:`K_i` with scalar prefactors :code:`coef`) :math:`c_i`,
    where the channel act on density matrix as :math:`\\rho'  = \sum_i K_i^\dagger \\rho K_i`.
    The data should satisfy :math:`\sum K_i^\dagger K_i = I`

    Attributes
    ----------
    nqubit : int
        number of qubits acted on by the Kraus operators
    size : int
        number of Kraus operators (== Choi rank)
    kraus_ops : array_like(dict())
        the data in format
        array_like(dict): [{coef: scalar, operator: array_like}, {coef: scalar, operator: array_like}, ...]

    Returns
    -------
    Channel object
        containing the corresponding Kraus operators

    """

    def __init__(self, kraus_data):
        """
        Parameters
        ----------
        kraus_data : array_like
            array of Kraus operator data.
            array_like(dict): [{coef: scalar, operator: array_like}, {parameter: scalar, operator: array_like}, ...]
            only works for square Kraus operators

        Raises
        ------
        ValueError
            If empty array_like is provided.
        """

        # check there is data
        if not kraus_data:
            raise ValueError("Cannot instantiate the channel with empty data.")

        if not isinstance(kraus_data, (list, np.ndarray, tuple)):
            raise TypeError(f"The data must be a list, a numpy.ndarray or a tuple not a {type(kraus_data)}.")

        # check that data is correctly formatted before assigning it to the object.
        assert check_data_values_type(kraus_data)
        assert check_data_dims(kraus_data)

        # check that the channel is properly normalized i.e
        # \sum_K_i^\dagger K_i = Identity
        assert check_data_normalization(kraus_data)
        assert check_rank(kraus_data)

        self.nqubit = int(np.log2(kraus_data[0]["operator"].shape[0]))
        self.kraus_ops = kraus_data

        # np.asarray(data, dtype=np.complex128)
        # number of Kraus operators in the Channel

        self.size = len(kraus_data)

    def __repr__(self):
        return f"KrausChannel object with {self.size} Kraus operators of dimension {self.nqubit}."

    def is_normalized(self):
        return check_data_normalization(self.kraus_ops)


def dephasing_channel(prob: float) -> KrausChannel:
    """single-qubit dephasing channel, :math:`(1-p) \\rho + p Z  \\rho Z`

    Parameters
    ----------
    prob : float
        The probability associated to the channel

    Returns
    -------
    :class:`graphix.channel.KrausChannel` object
        containing the corresponding Kraus operators
    """
    return KrausChannel(
        [{"coef": np.sqrt(1 - prob), "operator": np.eye(2)}, {"coef": np.sqrt(prob), "operator": Ops.z}]
    )


def depolarising_channel(prob: float) -> KrausChannel:
    """single-qubit depolarizing channel

    .. math::
        (1-p) \\rho + \\frac{p}{3} (X \\rho X + Y \\rho Y + Z \\rho Z) = (1 - 4 \\frac{p}{3}) \\rho + 4 \\frac{p}{3} id

    Parameters
    ----------
    prob : float
        The probability associated to the channel

    """
    return KrausChannel(
        [
            {"coef": np.sqrt(1 - prob), "operator": np.eye(2)},
            {"coef": np.sqrt(prob / 3.0), "operator": Ops.x},
            {"coef": np.sqrt(prob / 3.0), "operator": Ops.y},
            {"coef": np.sqrt(prob / 3.0), "operator": Ops.z},
        ]
    )


def pauli_channel(px: float, py: float, pz: float) -> KrausChannel:
    """single-qubit pauli channel,

    .. math::
        (1-p_X-p_Y-p_Z) \\rho + p_X X \\rho X + p_Y Y \\rho Y + p_Z Z \\rho Z)

    """
    if px + py + pz > 1:
        raise ValueError("The sum of probabilities must not exceed 1.")
    pI = 1 - px - py - pz
    return KrausChannel(
        [
            {"coef": np.sqrt(1 - pI), "operator": np.eye(2)},
            {"coef": np.sqrt(px / 3.0), "operator": Ops.x},
            {"coef": np.sqrt(py / 3.0), "operator": Ops.y},
            {"coef": np.sqrt(pz / 3.0), "operator": Ops.z},
        ]
    )


def two_qubit_depolarising_channel(prob: float) -> KrausChannel:
    """two-qubit depolarising channel.

    .. math::
        \mathcal{E} (\\rho) = (1-p) \\rho + \\frac{p}{15}  \sum_{P_i \in \{id, X, Y ,Z\}^{\otimes 2}/(id \otimes id)}P_i \\rho P_i

    Parameters
    ----------
    prob : float
        The probability associated to the channel

    Returns
    -------
    :class:`graphix.channel.KrausChannel` object
        containing the corresponding Kraus operators
    """

    return KrausChannel(
        [
            {"coef": np.sqrt(1 - prob), "operator": np.kron(np.eye(2), np.eye(2))},
            {"coef": np.sqrt(prob / 15.0), "operator": np.kron(Ops.x, Ops.x)},
            {"coef": np.sqrt(prob / 15.0), "operator": np.kron(Ops.z, Ops.z)},
            {"coef": np.sqrt(prob / 15.0), "operator": np.kron(Ops.x, np.eye(2))},
            {"coef": np.sqrt(prob / 15.0), "operator": np.kron(Ops.y, Ops.y)},
            {"coef": np.sqrt(prob / 15.0), "operator": np.kron(Ops.y, np.eye(2))},
            {"coef": np.sqrt(prob / 15.0), "operator": np.kron(Ops.z, np.eye(2))},
            {"coef": np.sqrt(prob / 15.0), "operator": np.kron(np.eye(2), Ops.x)},
            {"coef": np.sqrt(prob / 15.0), "operator": np.kron(np.eye(2), Ops.y)},
            {"coef": np.sqrt(prob / 15.0), "operator": np.kron(np.eye(2), Ops.z)},
            {"coef": np.sqrt(prob / 15.0), "operator": np.kron(Ops.x, Ops.y)},
            {"coef": np.sqrt(prob / 15.0), "operator": np.kron(Ops.x, Ops.z)},
            {"coef": np.sqrt(prob / 15.0), "operator": np.kron(Ops.y, Ops.x)},
            {"coef": np.sqrt(prob / 15.0), "operator": np.kron(Ops.y, Ops.z)},
            {"coef": np.sqrt(prob / 15.0), "operator": np.kron(Ops.z, Ops.y)},
            {"coef": np.sqrt(prob / 15.0), "operator": np.kron(Ops.z, Ops.x)},
        ]
    )


def two_qubit_depolarising_tensor_channel(prob: float) -> KrausChannel:
    """two-qubit tensor channel of single-qubit depolarising channels with same probability.
    Kraus operators:

    .. math::
        \Big\{ \sqrt{(1-p)} id, \sqrt{(p/3)} X, \sqrt{(p/3)} Y , \sqrt{(p/3)} Z \Big\} \otimes \Big\{ \sqrt{(1-p)} id, \sqrt{(p/3)} X, \sqrt{(p/3)} Y , \sqrt{(p/3)} Z \Big\}

    Parameters
    ----------
    prob : float
        The probability associated to the channel

    Returns
    -------
    :class:`graphix.channel.KrausChannel` object
        containing the corresponding Kraus operators
    """

    return KrausChannel(
        [
            {"coef": 1 - prob, "operator": np.kron(np.eye(2), np.eye(2))},
            {"coef": prob / 3.0, "operator": np.kron(Ops.x, Ops.x)},
            {"coef": prob / 3.0, "operator": np.kron(Ops.y, Ops.y)},
            {"coef": prob / 3.0, "operator": np.kron(Ops.z, Ops.z)},
            {"coef": np.sqrt(1 - prob) * np.sqrt(prob / 3.0), "operator": np.kron(Ops.x, np.eye(2))},
            {"coef": np.sqrt(1 - prob) * np.sqrt(prob / 3.0), "operator": np.kron(Ops.y, np.eye(2))},
            {"coef": np.sqrt(1 - prob) * np.sqrt(prob / 3.0), "operator": np.kron(Ops.z, np.eye(2))},
            {"coef": np.sqrt(1 - prob) * np.sqrt(prob / 3.0), "operator": np.kron(np.eye(2), Ops.x)},
            {"coef": np.sqrt(1 - prob) * np.sqrt(prob / 3.0), "operator": np.kron(np.eye(2), Ops.y)},
            {"coef": np.sqrt(1 - prob) * np.sqrt(prob / 3.0), "operator": np.kron(np.eye(2), Ops.z)},
            {"coef": prob / 3.0, "operator": np.kron(Ops.x, Ops.y)},
            {"coef": prob / 3.0, "operator": np.kron(Ops.x, Ops.z)},
            {"coef": prob / 3.0, "operator": np.kron(Ops.y, Ops.x)},
            {"coef": prob / 3.0, "operator": np.kron(Ops.y, Ops.z)},
            {"coef": prob / 3.0, "operator": np.kron(Ops.z, Ops.x)},
            {"coef": prob / 3.0, "operator": np.kron(Ops.z, Ops.y)},
        ]
    )
