import numpy as np

from graphix.Checks.generic_checks import check_square, check_psd
from typing import Union

# from graphix.kraus import Channel
# from graphix.sim.density_matrix import DensityMatrix


def check_data_normalization(data: Union[list, tuple, np.ndarray]) -> bool:
    # NOTE use np.conjugate() instead of object.conj() to certify behaviour when using non-numpy float/complex types
    opsu = np.array([i["parameter"] * np.conj(i["parameter"]) * i["operator"].conj().T @ i["operator"] for i in data])

    if not np.allclose(np.sum(opsu, axis=0), np.eye(2 ** int(np.log2(len(data[0]["operator"]))))):
        raise ValueError(f"The specified channel is not normalized. {np.sum(opsu, axis=0)}")
    return True


def check_data_dims(data: Union[list, tuple, np.ndarray]) -> bool:

    # convert to set to remove duplicates
    dims = list(set([i["operator"].shape for i in data]))
    # or list({[i["operator"].shape for i in data]}) using set comprehension

    # check all the same dimensions and that they are square matrices
    # TODO replace by using array.ndim
    if len(dims) != 1:
        raise ValueError(f"All provided Kraus operators do not have the same dimension {dims}!")

    # NOTE need an assert here???
    check_square(data[0]["operator"])

    # if dims[0][0] != dims[0][1]:
    #     raise ValueError(f"All provided Kraus operators have the same shape {dims[0]} but are not square matrices!")

    # # check consistency with tensor of qubit local Hilbert spaces
    # data_dim = np.log2(dims[0][0])
    # if not np.isclose(data_dim, int(data_dim)):
    #     raise ValueError(f"Incorrect data dimension {data_dim}: not consistent with qubits.")
    # # data_dim = int(data_dim)

    return True


def check_data_values_type(data: Union[list, tuple, np.ndarray]) -> bool:

    value_types = list(set([isinstance(i, dict) for i in data]))

    if value_types == [True]:

        key0_values = list(set([list(i.keys())[0] == "parameter" for i in data]))
        key1_values = list(set([list(i.keys())[1] == "operator" for i in data]))

        if key0_values == [True] and key1_values == [True]:
            operator_types = list(set([isinstance(i["operator"], np.ndarray) for i in data]))

            if operator_types == [True]:
                operator_dtypes = list(
                    set([i["operator"].dtype in [float, complex, np.float64, np.complex128] for i in data])
                )

                if operator_dtypes == [True]:
                    par_types = list(
                        set([isinstance(i["parameter"], (float, complex, np.float64, np.complex128)) for i in data])
                    )

                    if par_types == [True]:
                        pass
                    else:
                        raise TypeError("All parameters are not scalars")

                else:
                    raise TypeError(
                        f"All operators  {list([i['operator'].dtype == (float or complex or np.float64 or np.complex128) for i in data])}."
                    )
            else:
                raise TypeError("All operators don't have the same type and must be np.ndarray.")
        else:
            raise KeyError("The keys of the indivudal Kraus operators must be parameter and operator.")
    else:
        raise TypeError("All values are not dictionaries.")

    return True


def check_rank(data: Union[list, tuple, np.ndarray]) -> bool:
    # already checked that the data is square matrix
    if len(data) > data[0]["operator"].shape[0] ** 2:
        raise ValueError(
            f"Incorrect number of Kraus operators in the expansion. This number must be an integer between 1 and the dimension squared."
        )

    return True


# TODO move that to kraus.py

# def build_choi(channel: Channel):
#     """Method to build the Choi state associated to channel :math: `\mathcal{E}`
#     as :math:`\mathds{1}_R \otimes \mathcal{E}`
#     TODO recall calculation convention.
#     Doesn't matter if column- or row-stacking convention since only vectorizing the identity.

#     Args:
#         channel (:class: `graphix.kraus.Channel`): channel from which to build the Choi state.

#     Returns:
#         :class: `graphix.sim.density_matrix.DensityMatrix`: Choi state.
#     """
#     dim = 2**channel.nqubit
#     # build max ent state
#     # TODO do a subroutine somewhere? In Channel claas? In the same file?
#     maxent_state = np.eye(dim).ravel()

#     # build associated density matrix as DensityMatrix object
#     maxent_state /= np.sqrt(np.sum(np.abs(maxent_state) ** 2))
#     maxent_state_dm = DensityMatrix(data=np.outer(maxent_state, maxent_state.conj()))

#     # apply channel on second half of the Hilbert space
#     Choi_state = maxent_state_dm.apply_channel(channel, list(range(channel.nqubit, 2 * channel.nqubit)))

#     return Choi_state


# def check_cp(channel: Channel) -> bool:
#     """Method to check the complete positivity (CP) of the Channel using the Choi state :math:`\mathds{1}_R \otimes \mathcal{E}`
#     Args:
#          channel (:class: `graphix.kraus.Channel`): channel to test

#     Returns:
#         bool: True if the Channel is CP. Else check_psd raises a ValueError.
#     """

#     Choi_state = build_choi(channel)
#     # TODO make a method to extend all channels to double Hilbert space
#     assert check_psd(Choi_state)

#     return True
