import numpy as np


def to_kraus(data):
    r"""Convert input data into Kraus operator set.

    Parameters
    ----------
        data : array_like
            Data to convert into Kraus operator set.
            Input data must be either (i)single Operator, (ii)single Kraus set or (iii)generalized Kraus set.
            Relation among quantum channel, input data and returned Kraus operator set is as follwos:
                (i) quantum channel: :math:`E(\rho) = A \rho A^\dagger`
                    input data: A (2d-array-like)
                    returned Kraus operator set: (A, None)
                (ii) quantum channel: :math:`E(\rho) = \sum_i A_i \rho A_i^\dagger`
                    input data: [A_1, A_2, ...]
                    returned Kraus operator set: ([A_1, A_2, ...], None)
                (iii) quantum channel: :math:`E(\rho) = \sum_i A_i \rho B_i^\dagger`
                    input data: [[A_1, A_2, ...], [B_1, B_2, ...]]
                    returned Kraus operator set: ([A_1, A_2, ...], [B_1, B_2, ...])
    Returns
    -------
        kraus : tuple
            Kraus operator set.
    """
    if isinstance(data, (list, tuple, np.ndarray)):
        # (i) If input is 2d-array-like, it is a single unitary matrix A for channel:
        # E(rho) = A * rho * A^\dagger
        if _is_matrix(data):
            return ([np.asarray(data, dtype=complex)], None)

        # (ii) If input is list of 2d-array-likes, it is a single Kraus set for channel:
        # E(rho) = \sum_i A_i * rho * A_i^\dagger
        elif isinstance(data, (list, tuple, np.ndarray)) and len(data) > 0 and _is_matrix(data[0]):
            if isinstance(data, np.ndarray):
                data = data.tolist()
            kraus = [np.asarray(data[0], dtype=complex)]
            shape = kraus[0].shape
            for A_i in data[1:]:
                A_i = np.asarray(A_i, dtype=complex)
                if A_i.shape != shape:
                    raise ValueError("All Kraus operators must have same shape.")
                kraus.append(A_i)
            return (kraus, None)

        # (iii) If input is list of lists consisting of same number of 2d-array-likes,
        # it is a generalized Kraus set for channel:
        # E(rho) = \sum_i A_i * rho * B_i^\dagger
        elif (
            isinstance(data, (list, tuple, np.ndarray))
            and len(data) == 2
            and len(data[0]) == len(data[1])
            and len(data[0]) > 0
        ):
            if isinstance(data, np.ndarray):
                data = data.tolist()
            if not _is_matrix(data[0][0]):
                raise ValueError(
                    "Input data must be either (i)single Operator (2d-array-like), "
                    "(ii)single Kraus set (list of 2d-array-likes) or "
                    "(iii)generalized Kraus set (list of lists consisting of same number of 2d-array-likes)."
                )
            kraus_left = [np.asarray(data[0][0], dtype=complex)]
            shape = kraus_left[0].shape
            for A_i in data[0][1:]:
                A_i = np.asarray(A_i, dtype=complex)
                if A_i.shape != shape:
                    raise ValueError("All Kraus operators must have same shape.")
                kraus_left.append(A_i)
            kraus_right = []
            for B_i in data[1]:
                B_i = np.asarray(B_i, dtype=complex)
                if B_i.shape != shape:
                    raise ValueError("All Kraus operators must have same shape.")
                kraus_right.append(B_i)
            return (kraus_left, kraus_right)

        else:
            raise ValueError(
                "Input data must be either (i)single Operator (2d-array-like), "
                "(ii)single Kraus set (list of 2d-array-likes) or "
                "(iii)generalized Kraus set (list of lists consisting of same number of 2d-array-likes)."
            )
    else:
        raise TypeError("Input data must be list, tupple, or array_like.")


def generate_depolarizing_kraus(p, nqubits):
    """Return Kraus operators for a depolarizing channel."""
    pass


def generate_amplitude_damping_kraus(p, nqubits):
    """Return Kraus operators for an amplitude damping channel."""
    pass


def generate_dephasing_kraus(p, nqubits):
    """Return Kraus operators for a dephasing channel."""
    pass


def _is_matrix(data):
    """Check if data is a matrix."""
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    return data.shape == (2, 2)
