import numpy as np
from typing import Union


def check_square(matrix: np.ndarray) -> bool:
    """
    check if matrix is a square matrix with a power of 2 dimension.
    """

    if len(matrix.shape) != 2:
        raise ValueError(f"The object has {len(matrix.shape)} axes but must have 2 to be a matrix.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Matrix must be square but has different dimensions {matrix.shape}.")
    size = matrix.shape[0]
    if size & (size - 1) != 0:
        raise ValueError(f"Matrix size must be a power of two but is {size}.")
    return True


def check_psd(matrix: np.ndarray, tol: float = 1e-15) -> bool:
    """
    check if a density matrix is positive semidefinite by diagonalizing.
    After check_square and check_hermitian (osef) so that it already is square with power of 2 dimension.


    Parameters
    ----------
    matrix : np.ndarray
        matrix to check. Normally already square and 2**n x 2**n
    tol : float
        tolerance on the small negatives. Default 1e-15.
    method : 'choleski' or 'sylvester' (or 'eigendecomp'). Default 'choleski'
        if method = 'choleski': attempts a Choleski decomposition. If the matrix is not PSD raises a numpy.linalg.LinAlgError. (but error can be something else...)
        if method = 'sylvester': computes the determinants of all square matrices of increasing dimension starting from the top left corner. Sylvester : PSD <-> all principal minors are non negative. NOT just leading!!!!!!
        NOPE. Sylvester for PSD is all principal minors, not just leading !!! So VERY expensive.
    """

    # if matrix is not hermitian, raises an error
    # if no error: eigenvals are real

    evals = np.linalg.eigvalsh(matrix)
    print(matrix)
    print(evals)
    # remove small negatives like -1e-17
    evals[np.abs(evals) < tol] = 0

    # sort (ascending order)
    evals.sort()

    # PSD test. Just look
    if not evals[0] >= 0:
        raise ValueError("The matrix is not positive semi-definite.")

    return True


def check_hermitian(matrix: np.ndarray) -> bool:
    """
    check if matrix is hermitian. After check_square.
    """

    if not np.allclose(matrix, matrix.transpose().conjugate()):
        raise ValueError("The matrix is not Hermitian.")
    return True


def check_unit_trace(matrix: np.ndarray) -> bool:
    """
    check if matrix has trace 1. After check_square.
    """

    if not np.allclose(matrix.trace(), 1.0):
        raise ValueError("The matrix does not have unit trace.")
    return True


def check_data_normalization(data: Union[list, tuple, np.ndarray]) -> bool:
    # NOTE use np.conjugate() instead of object.conj() to certify behaviour when using non-numpy float/complex types
    opsu = np.array([i["coef"] * np.conj(i["coef"]) * i["operator"].conj().T @ i["operator"] for i in data])

    if not np.allclose(np.sum(opsu, axis=0), np.eye(2 ** int(np.log2(len(data[0]["operator"]))))):
        raise ValueError(f"The specified channel is not normalized. {np.sum(opsu, axis=0)}")
    return True


def check_data_dims(data: Union[list, tuple, np.ndarray]) -> bool:

    # convert to set to remove duplicates
    dims = list(set([i["operator"].shape for i in data]))

    # check all the same dimensions and that they are square matrices
    # TODO replace by using array.ndim
    if len(dims) != 1:
        raise ValueError(f"All provided Kraus operators do not have the same dimension {dims}!")

    assert check_square(data[0]["operator"])

    return True


def check_data_values_type(data: Union[list, tuple, np.ndarray]) -> bool:

    value_types = list(set([isinstance(i, dict) for i in data]))

    if value_types == [True]:

        key0_values = list(set([list(i.keys())[0] == "coef" for i in data]))
        key1_values = list(set([list(i.keys())[1] == "operator" for i in data]))

        if key0_values == [True] and key1_values == [True]:
            operator_types = list(set([isinstance(i["operator"], np.ndarray) for i in data]))

            if operator_types == [True]:
                operator_dtypes = list(
                    set([i["operator"].dtype in [float, complex, np.float64, np.complex128] for i in data])
                )

                if operator_dtypes == [True]:
                    par_types = list(
                        set([isinstance(i["coef"], (float, complex, np.float64, np.complex128)) for i in data])
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
            raise KeyError("The keys of the indivudal Kraus operators must be coef and operator.")
    else:
        raise TypeError("All values are not dictionaries.")

    return True


def check_rank(data: Union[list, tuple, np.ndarray]) -> bool:
    # already checked that the data is list of square matrices
    if len(data) > data[0]["operator"].shape[0] ** 2:
        raise ValueError(
            f"Incorrect number of Kraus operators in the expansion. This number must be an integer between 1 and the dimension squared."
        )

    return True
