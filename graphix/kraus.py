import numpy as np


class Channel:
    """(Noise) Channel class in the Kraus representation

    Parameters
    ----------
        data : array_like
            array of Kraus operator data. dict(dict): {0: {parameter: float, operator: array_like}, 1: {parameter: float, operator: array_like}, ...}
    """

    # TODO json compatibility and allow to import channels from file?
    # TODO ? or *data and build from several (parameter, operator) couples?
    def __init__(self, kraus_data):

        # check there is data
        assert kraus_data
        assert isinstance(kraus_data, dict)
        assert len(kraus_data) >= 1

        # check that data is correctly formatted before assigning it to the object. 
        # Maybe not the best way to do it?
        # Raises an error if something is incorrect.
        # remove if do conversion later.
        check_data_values_type(kraus_data)
        # raise TypeError("Incorrect types and formats")
        # TODO also add the check that keys are "parameter" and "value". Do that behind the scenes.

        # don't use assert to validate data! https://www.pythoniste.fr/python/linstruction-dassertion-en-python-assert/
        # better?
        # test!
        # if not check_data_dims(kraus_data):
        #     raise ValueError("All provided Kraus operators don't have the same dimension!")

        check_data_dims(kraus_data)

        # add an euseless since there's an error raised?
        # TODO ask TM
        # else: # non empty and all same dim
        self.nqubit = int(np.log2(kraus_data[0]["operator"].shape[0]))
        self.kraus_ops = kraus_data

        # np.asarray(data, dtype=np.complex128)
        # number of Kraus operators in the Channel
        self.size = len(kraus_data)

    # TODO update
    def __repr__(self):
        return f"Channel object with (data={self.kraus_ops}, qarg={self.qarg})"


def check_data_dims(data):

    result = False
    # convert to set to remove duplicates
    dims = list(set([i["operator"].shape for i in data.values()]))
    # check all the same dimensions and that they are square matrices
    if len(dims) == 1 and len(list(dims)[0]) == 2 and dims[0][0] == dims[0][1]:
        result = True
    else:
        raise ValueError("All provided Kraus operators don't have the same dimension!")

    return result


def check_data_values_type(data):
    # convert to set to remove duplicates

    # TODO put error raising here instead. And deaggregate this mess to raise useful errors.

    result = False
    # also check the values in the arrays !!!
    value_types = list(set([isinstance(i, dict) for i in data.values()]))
    

    if value_types == [True]:

        key0_values = list(set([list(i.keys())[0] =='parameter' for i in data.values()]))
        key1_values = list(set([list(i.keys())[1] =='operator' for i in data.values()]))

        if key0_values == [True] and key1_values == [True]: 
            operator_types = list(set([isinstance(i["operator"], np.ndarray) for i in data.values()]))

            if operator_types == [True]:
                operator_dtypes = list(set([i["operator"].dtype == (float or complex or np.float64 or np.complex128)  for i in data.values()]))
                    
                if operator_dtypes == [True]:
                    par_types = list(set([isinstance(i["parameter"], (float, complex, np.float64, np.complex128)) for i in data.values()]))


                    if par_types == [True] :
                        result = True
                    else:
                        raise TypeError("All parameters are not scalars")

                else: 
                    raise TypeError("All operators don't have the same dtype.")

            else: 
                raise TypeError("All operators don't have the same type.")
        else:
            raise KeyError("The keys of the indivudal Kraus operators must be parameter and operator.")
    else:
        raise TypeError("All values are not dictionaries.")
    
    return result

def check_data_normalization(data):
    # convert to set to remove duplicates
    # TODO implement
    pass

# # maybe later if not dict of dict but array_like of array_like
# def to_kraus(data):
#     r"""Convert input data into Kraus operator set [KrausOp, KrausOp, ...].
#     Each KrausOp has unitary matrix and target qubit index info.

#     Parameters
#     ----------
#         data : array_like
#             Data to convert into Kraus operator set.
#             Input data must be either (i)single Operator or (ii)Kraus set.
#             Relation among quantum channel, input data and returned Kraus operator set is as follwos:
#                 (i) quantum channel: :math:`E(\rho) = A \rho A^\dagger`
#                     input data: [A (2d-array-like), qarg(int)]
#                     returns: [KrausOp]
#                 (ii) quantum channel: :math:`E(\rho) = \sum_i A_i \rho A_i^\dagger`
#                     input data: [(A_1, int), (A_2, int), ...]
#                     returns: [KrausOp, KrausOp, ...]
#     Returns
#     -------
#         kraus : list. [KrausOp, ...]
#             KrausOp set.
#     """
#     if isinstance(data, (list, tuple, np.ndarray)):
#         if len(data) <= 1:
#             raise ValueError(
#                 "Input data must be either single Kraus Operator, single Kraus set or generalized Kraus set"
#                 " with target qubit indices."
#             )
#         # (i) If input is [2d-array-like, int], the first data is a single unitary matrix A for channel:
#         # E(rho) = A * rho * A^\dagger
#         # and the second data is target qubit index.
#         if _is_kraus_op(data):
#             return [KrausOp(data=data[0], qarg=data[1])]

#         # (ii) If input is list of [2d-array-likes, int], it is a single Kraus set for channel:
#         # E(rho) = \sum_i A_i * rho * A_i^\dagger
#         # with target qubit indices.
#         elif isinstance(data, (list, tuple, np.ndarray)) and _is_kraus_op(data[0]):
#             if isinstance(data, np.ndarray):
#                 data = data.tolist()
#             kraus = [KrausOp(data=data[0][0], qarg=data[0][1])]
#             for A_i in data[1:]:
#                 A_i = KrausOp(data=A_i[0], qarg=A_i[1])
#                 if _is_kraus_op(A_i):
#                     raise ValueError("All Kraus operators must have same shape.")
#                 kraus.append(A_i)
#             return kraus
#         else:
#             raise ValueError(
#                 "Input data must be either (i)single Operator (2d-array-like)"
#                 " or (ii)single Kraus set (list of 2d-array-likes)"
#                 " with qubit indices for each Operator."
#             )
#     else:
#         raise TypeError("Input data must be list, tupple, or array_like.")


# def generate_dephasing_kraus(p, qarg):
#     """Return Kraus operators for a dephasing channel.

#     Parameters
#     ----------
#         p : float
#             Probability of dephasing error.
#         qarg : int
#             Target qubit index.
#     """
#     assert isinstance(qarg, int)
#     assert 0 <= p <= 1
#     return to_kraus([[np.sqrt(1 - p) * np.eye(2), qarg], [np.sqrt(p) * np.diag([1, -1]), qarg]])


# def generate_depolarizing_kraus(p, nqubits):
#     """Return Kraus operators for a depolarizing channel."""
#     pass


# def generate_amplitude_damping_kraus(p, nqubits):
#     """Return Kraus operators for an amplitude damping channel."""
#     pass


# def _is_kraus_op(data):
#     """Check if data is a Kraus operator.
#     Currently, Kraus operator is defined as a list of [2d-array-like, int].
#     This might be changed in the future to support Kraus operator of the form [2^k dim array-like, int] (k <= n).
#     """
#     if not isinstance(data, (list, tuple, np.ndarray)):
#         return False
#     if len(data) != 2:
#         return False
#     if not isinstance(data[1], int):
#         return False
#     if not isinstance(data[0], np.ndarray):
#         return np.array(data[0]).shape == (2, 2)
#     return data[0].shape == (2, 2)
