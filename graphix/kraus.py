import numpy as np


class KrausOp:
    """Kraus operator class.

    Parameters
    ----------
        data : array_like
            Kraus operator data. 2d-array-like.
            This might be changed in the future to support Kraus operator of 2^k dim array-like (k <= n).
        qarg : int
            Target qubit index.
    """

    def __init__(self, data, qarg):
        assert isinstance(qarg, int)
        self.data = np.asarray(data, dtype=complex)
        self.qarg = qarg

    def __repr__(self):
        return f"KrausOp(data={self.data}, qarg={self.qarg})"


def to_kraus(data):
    r"""Convert input data into Kraus operator set [KrausOp, KrausOp, ...].
    Each KrausOp has unitary matrix and target qubit index info.

    Parameters
    ----------
        data : array_like
            Data to convert into Kraus operator set.
            Input data must be either (i)single Operator or (ii)Kraus set.
            Relation among quantum channel, input data and returned Kraus operator set is as follwos:
                (i) quantum channel: :math:`E(\rho) = A \rho A^\dagger`
                    input data: [A (2d-array-like), qarg(int)]
                    returns: [KrausOp]
                (ii) quantum channel: :math:`E(\rho) = \sum_i A_i \rho A_i^\dagger`
                    input data: [(A_1, int), (A_2, int), ...]
                    returns: [KrausOp, KrausOp, ...]
    Returns
    -------
        kraus : list. [KrausOp, ...]
            KrausOp set.
    """
    if isinstance(data, (list, tuple, np.ndarray)):
        if len(data) <= 1:
            raise ValueError(
                "Input data must be either single Kraus Operator, single Kraus set or generalized Kraus set"
                " with target qubit indices."
            )
        # (i) If input is [2d-array-like, int], the first data is a single unitary matrix A for channel:
        # E(rho) = A * rho * A^\dagger
        # and the second data is target qubit index.
        if _is_kraus_op(data):
            return [KrausOp(data=data[0], qarg=data[1])]

        # (ii) If input is list of [2d-array-likes, int], it is a single Kraus set for channel:
        # E(rho) = \sum_i A_i * rho * A_i^\dagger
        # with target qubit indices.
        elif isinstance(data, (list, tuple, np.ndarray)) and _is_kraus_op(data[0]):
            if isinstance(data, np.ndarray):
                data = data.tolist()
            kraus = [KrausOp(data=data[0][0], qarg=data[0][1])]
            for A_i in data[1:]:
                A_i = KrausOp(data=A_i[0], qarg=A_i[1])
                if _is_kraus_op(A_i):
                    raise ValueError("All Kraus operators must have same shape.")
                kraus.append(A_i)
            return kraus
        else:
            raise ValueError(
                "Input data must be either (i)single Operator (2d-array-like)"
                " or (ii)single Kraus set (list of 2d-array-likes)"
                " with qubit indices for each Operator."
            )
    else:
        raise TypeError("Input data must be list, tupple, or array_like.")


def generate_dephasing_kraus(p, qarg):
    """Return Kraus operators for a dephasing channel.

    Parameters
    ----------
        p : float
            Probability of dephasing error.
        qarg : int
            Target qubit index.
    """
    assert isinstance(qarg, int)
    assert 0 <= p <= 1
    return [KrausOp(data=np.sqrt(1 - p) * np.eye(2), qarg=qarg), KrausOp(data=np.sqrt(p) * np.diag([1, -1]), qarg=qarg)]


def generate_depolarizing_kraus(p, nqubits):
    """Return Kraus operators for a depolarizing channel."""
    pass


def generate_amplitude_damping_kraus(p, nqubits):
    """Return Kraus operators for an amplitude damping channel."""
    pass


def _is_kraus_op(data):
    """Check if data is a Kraus operator.
    Currently, Kraus operator is defined as a list of [2d-array-like, int].
    This might be changed in the future to support Kraus operator of the form [2^k dim array-like, int] (k <= n).
    """
    if not isinstance(data, (list, tuple, np.ndarray)):
        return False
    if len(data) != 2:
        return False
    if not isinstance(data[1], int):
        return False
    if not isinstance(data[0], np.ndarray):
        return np.array(data[0]).shape == (2, 2)
    return data[0].shape == (2, 2)
