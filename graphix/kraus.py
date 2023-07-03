import numpy as np


class KrausOp:
    def __init__(self, data, qarg):
        assert isinstance(qarg, int)
        self.data = np.asarray(data, dtype=complex)
        self.qarg = qarg


def to_kraus(data):
    r"""Convert input data into Kraus operator set.

    Parameters
    ----------
        data : array_like
            Data to convert into Kraus operator set.
            Input data must be either (i)single Operator, (ii)single Kraus set or (iii)generalized Kraus set.
            Relation among quantum channel, input data and returned Kraus operator set is as follwos:
                (i) quantum channel: :math:`E(\rho) = A \rho A^\dagger`
                    input data: [A (2d-array-like), qarg(int)]
                    returns: ([KrausOp], None)
                (ii) quantum channel: :math:`E(\rho) = \sum_i A_i \rho A_i^\dagger`
                    input data: [(A_1, int), (A_2, int), ...]
                    returns: ([KrausOp, KrausOp, ...], None)
                (iii) quantum channel: :math:`E(\rho) = \sum_i A_i \rho B_i^\dagger`
                    input data: [[(A_1, int), (A_2, int), ...], [(B_1, int), (B_2, int), ...]]
                    returns: ([KrausOp, KrausOp, ...], [KrausOp, KrausOp, ...])
    Returns
    -------
        kraus : tuple. ([KrausOp, ...], Optional[[KrausOp, ...]])
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
            return ([KrausOp(data[0], data[1])], None)

        # (ii) If input is list of [2d-array-likes, int], it is a single Kraus set for channel:
        # E(rho) = \sum_i A_i * rho * A_i^\dagger
        # with target qubit indices.
        elif isinstance(data, (list, tuple, np.ndarray)) and _is_kraus_op(data[0]):
            if isinstance(data, np.ndarray):
                data = data.tolist()
            kraus = [KrausOp(data[0][0], data[0][1])]
            shape = kraus[0].data.shape
            for A_i in data[1:]:
                A_i = KrausOp(A_i[0], A_i[1])
                if A_i.data.shape != shape:
                    raise ValueError("All Kraus operators must have same shape.")
                kraus.append(A_i)
            return (kraus, None)

        # (iii) If input is like [[[2d-array-likes, int], ...], [[2d-array-likes, int], ...]],
        # it is a generalized Kraus set for channel:
        # E(rho) = \sum_i A_i * rho * B_i^\dagger
        # with target qubit indices.
        elif isinstance(data, (list, tuple, np.ndarray)) and len(data) == 2:
            if isinstance(data, np.ndarray):
                data = data.tolist()
            if len(data[0]) == 0:
                raise ValueError("Input data must be non-empty.")
            if not _is_kraus_op(data[0][0]):
                raise ValueError(
                    "Input data must be either (i)single Operator (2d-array-like), "
                    "(ii)single Kraus set (list of 2d-array-likes) or "
                    "(iii)generalized Kraus set (list of lists consisting of same number of 2d-array-likes)."
                    " with qubit indices for each Operator."
                )
            kraus_left = [KrausOp(data[0][0][0], data[0][0][1])]
            for A_i in data[0][1:]:
                if not _is_kraus_op(A_i):
                    raise ValueError(
                        "Input data must be either (i)single Operator (2d-array-like), "
                        "(ii)single Kraus set (list of 2d-array-likes) or "
                        "(iii)generalized Kraus set (list of lists consisting of same number of 2d-array-likes)."
                        " with qubit indices for each Operator."
                    )
                kraus_left.append(KrausOp(A_i[0], A_i[1]))
            kraus_right = []
            for B_i in data[1]:
                if not _is_kraus_op(B_i):
                    raise ValueError(
                        "Input data must be either (i)single Operator (2d-array-like), "
                        "(ii)single Kraus set (list of 2d-array-likes) or "
                        "(iii)generalized Kraus set (list of lists consisting of same number of 2d-array-likes)."
                        " with qubit indices for each Operator."
                    )
                kraus_right.append(KrausOp(B_i[0], B_i[1]))
            return (kraus_left, kraus_right)

        else:
            raise ValueError(
                "Input data must be either (i)single Operator (2d-array-like), "
                "(ii)single Kraus set (list of 2d-array-likes) or "
                "(iii)generalized Kraus set (list of lists consisting of same number of 2d-array-likes)"
                " with qubit indices for each Operator."
            )
    else:
        raise TypeError("Input data must be list, tupple, or array_like.")


def generate_dephasing_kraus(p, nqubits):
    """Return Kraus operators for a dephasing channel."""
    pass


def generate_depolarizing_kraus(p, nqubits):
    """Return Kraus operators for a depolarizing channel."""
    pass


def generate_amplitude_damping_kraus(p, nqubits):
    """Return Kraus operators for an amplitude damping channel."""
    pass


def _is_kraus_op(data):
    """Check if data is a Kraus operator."""
    if not isinstance(data, (list, tuple, np.ndarray)):
        return False
    if len(data) != 2:
        return False
    if not isinstance(data[1], int):
        return False
    if not isinstance(data[0], np.ndarray):
        return np.array(data[0]).shape == (2, 2)
    return data[0].shape == (2, 2)
