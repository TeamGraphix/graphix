"""Performant module for linear algebra on GF2 field."""

from __future__ import annotations

from typing import TYPE_CHECKING

import galois
import numpy as np
from numba import njit

if TYPE_CHECKING:
    import galois.typing as gt
    import numpy.typing as npt


class MatGF2:
    """Matrix on GF2 field."""

    def __init__(self, data: gt.ElementLike | gt.ArrayLike | MatGF2) -> None:
        """Construct a matrix of GF2.

        Parameters
        ----------
        data : array_like
            Input data
        """
        if not isinstance(data, MatGF2):
            self.data = galois.GF2(data)
        else:
            self.data = data.data

    def __repr__(self) -> str:
        """Return the representation string of the matrix."""
        return repr(self.data)

    def __str__(self) -> str:
        """Return the displayable string of the matrix."""
        return str(self.data)

    def __eq__(self, other: object) -> bool:
        """Return `True` if two matrices are equal, `False` otherwise."""
        if not isinstance(other, MatGF2):
            return NotImplemented
        return bool(np.all(self.data == other.data))

    def __add__(self, other: gt.ElementLike | gt.ArrayLike | MatGF2) -> MatGF2:
        """Add two matrices."""
        if not isinstance(other, MatGF2):
            other = MatGF2(other)
        return MatGF2(self.data + other.data)

    def __sub__(self, other: gt.ElementLike | gt.ArrayLike | MatGF2) -> MatGF2:
        """Substract two matrices."""
        if not isinstance(other, MatGF2):
            other = MatGF2(other)
        return MatGF2(self.data - other.data)

    def __mul__(self, other: gt.ElementLike | gt.ArrayLike | MatGF2) -> MatGF2:
        """Compute the point-wise multiplication of two matrices."""
        if not isinstance(other, MatGF2):
            other = MatGF2(other)
        return MatGF2(self.data * other.data)

    def __matmul__(self, other: gt.ElementLike | gt.ArrayLike | MatGF2) -> MatGF2:
        """Multiply two matrices."""
        if not isinstance(other, MatGF2):
            other = MatGF2(other)
        return MatGF2(self.data @ other.data)

    def __getitem__(self, key) -> MatGF2:
        """Allow numpy-style slicing."""
        return MatGF2(self.data.__getitem__(key))

    def __setitem__(self, key, value: gt.ElementLike | gt.ArrayLike | MatGF2) -> None:
        """Assign new value to data field.

        Verification that `value` is a valid finite field element is done at the level of the `galois.GF2.__setitem__` method.
        """
        if isinstance(value, MatGF2):
            value = value.data
        self.data.__setitem__(key, value)

    def __bool__(self) -> bool:
        """Define truthiness of `MatGF2` following galois (and, therefore, numpy) style."""
        return self.data.__bool__()

    def copy(self) -> MatGF2:
        """Return a copy of the matrix."""
        return MatGF2(self.data.copy())

    def add_row(self, array_to_add: npt.NDArray[np.uint8] | None = None, row: int | None = None) -> None:
        """Add a row to the matrix.

        Parameters
        ----------
        array_to_add : array_like (optional)
            Row to add. Defaults to `None`. If `None`, add a zero row.
        row : int (optional)
            Index where the new row is added. Defaults to `None`. If `None`, row is added at the end of the matrix.
        """
        if row is None:
            row = self.data.shape[0]
        if array_to_add is None:
            array_to_add = np.zeros((1, self.data.shape[1]), dtype=int)
        array_to_add = array_to_add.reshape((1, self.data.shape[1]))
        self.data = galois.GF2(np.insert(self.data, row, array_to_add, axis=0))

    def add_col(self, array_to_add: npt.NDArray[np.uint8] | None = None, col: int | None = None) -> None:
        """Add a column to the matrix.

        Parameters
        ----------
        array_to_add : array_like (optional)
            Column to add. Defaults to `None`. If `None`, add a zero column.
        col : int (optional)
            Index where the new column is added. Defaults to `None`. If `None`, column is added at the end of the matrix.
        """
        if col is None:
            col = self.data.shape[1]
        if array_to_add is None:
            array_to_add = np.zeros((1, self.data.shape[0]), dtype=int)
        array_to_add = array_to_add.reshape((1, self.data.shape[0]))
        self.data = galois.GF2(np.insert(self.data, col, array_to_add, axis=1))

    def concatenate(self, other: MatGF2, axis: int = 1) -> None:
        """Concatenate two matrices.

        Parameters
        ----------
        other : MatGF2
            Matrix to concatenate.
        axis: int (optional)
            Axis along which concatenate. Defaults to 1.
        """
        self.data = galois.GF2(np.concatenate((self.data, other.data), axis=axis))

    def remove_row(self, row: int) -> None:
        """Remove a row from the matrix.

        Parameters
        ----------
        row : int
            Index of row to be removed.
        """
        self.data = galois.GF2(np.delete(self.data, row, axis=0))

    def remove_col(self, col: int) -> None:
        """Remove a column from the matrix.

        Parameters
        ----------
        col : int
            Index of column to be removed.
        """
        self.data = galois.GF2(np.delete(self.data, col, axis=1))

    def swap_row(self, row1: int, row2: int) -> None:
        """Swap two rows.

        Parameters
        ----------
        row1 : int
            Row index.
        row2 : int
            Row index.
        """
        self.data[[row1, row2]] = self.data[[row2, row1]]

    def swap_col(self, col1: int, col2: int) -> None:
        """Swap two columns.

        Parameters
        ----------
        col1: int
            Column index.
        col2 : int
            Column index.
        """
        self.data[:, [col1, col2]] = self.data[:, [col2, col1]]

    def permute_row(self, row_permutation: npt.ArrayLike) -> None:
        """Permute rows.

        Parameters
        ----------
        row_permutation : array_like
            Row permutation.
        """
        self.data = self.data[row_permutation, :]

    def permute_col(self, col_permutation: npt.ArrayLike) -> None:
        """Permute columns.

        Parameters
        ----------
        col_permutation : array_like
            Column permutation
        """
        self.data = self.data[:, col_permutation]

    def get_rank(self) -> int:
        """Get the rank of the matrix.

        Returns
        -------
        int : int
            Rank of the matrix.
        """
        mat_a = self.row_reduce()
        return int(np.sum(mat_a.data.any(axis=1)))

    def right_inverse(self) -> MatGF2 | None:
        r"""Return any right inverse of the matrix.

        Returns
        -------
        rinv : MatGF2
            Any right inverse of the matrix.
        or `None`
            If the matrix does not have a right inverse.

        Notes
        -----
        Let us consider a matrix :math:`A` of size :math:`(m \times n)`. The right inverse is a matrix :math:`B` of size :math:`(n \times m)` s.t. :math:`AB = I` where :math:`I` is the identity matrix.
        - The right inverse only exists if :math:`rank(A) = m`. Therefore, it is necessary but not sufficient that :math:`m ≤ n`.
        - The right inverse is unique only if :math:`m=n`.
        """
        m, n = self.data.shape
        if m > n:
            return None

        ident = galois.GF2.Identity(m)
        aug = galois.GF2(np.hstack([self.data, ident]))
        red = MatGF2(aug).row_reduce(ncols=n).data  # Reduced row echelon form

        # Check that rank of right block is equal to the number of rows.
        # We don't use `MatGF2.get_rank()` to avoid row-reducing twice.
        if m != int(np.sum(red[:, :n].any(axis=1))):
            return None
        rinv = galois.GF2.Zeros((n, m))

        for i, row in enumerate(red):
            j = np.flatnonzero(row)[0]  # Column index corresponding to the leading 1 in row `i`.
            rinv[j, :] = red[i, n:]

        return MatGF2(rinv)

    def null_space(self) -> MatGF2:
        r"""Return the null space of the matrix.

        Returns
        -------
        MatGF2
            The rows of the basis matrix are the basis vectors that span the null space. The number of rows of the basis matrix is the dimension of the null space.

        Notes
        -----
        This implementation appear to be more efficient than `:func:galois.GF2.null_space`.
        """
        m, n = self.data.shape

        ident = galois.GF2.Identity(n)
        ref = MatGF2(galois.GF2(np.hstack([self.data.T, ident])))
        ref.gauss_elimination(ncols=m)
        row_idxs = np.flatnonzero(
            ~ref.data[:, :m].any(axis=1)
        )  # Row indices of the 0-rows in the first block of `ref`.

        return ref[row_idxs, m:]

    def transpose(self) -> MatGF2:
        r"""Return transpose of the matrix."""
        return MatGF2(self.data.T)

    def gauss_elimination(self, ncols: int | None = None, copy: bool = False) -> MatGF2:
        """Return row echelon form (REF) by performing Gaussian elimination.

        Parameters
        ----------
        n_cols : int (optional)
            Number of columns over which to perform Gaussian elimination. The default is `None` which represents the number of columns of the matrix.

        copy : bool (optional)
            If `True`, the REF matrix is copied into a new instance, otherwise `self` is modified. Defaults to `False`.

        Returns
        -------
        mat_ref : MatGF2
            The matrix in row echelon form.
        """
        ncols = self.data.shape[1] if ncols is None else ncols
        mat_ref = MatGF2(self.data) if copy else self

        return MatGF2(_elimination_jit(mat_ref.data, ncols=ncols, full_reduce=False))

    def row_reduce(self, ncols: int | None = None, copy: bool = False) -> MatGF2:
        """Return row-reduced echelon form (RREF) by performing Gaussian elimination.

        Parameters
        ----------
        n_cols : int (optional)
            Number of columns over which to perform Gaussian elimination. The default is `None` which represents the number of columns of the matrix.

        copy : bool (optional)
            If `True`, the RREF matrix is copied into a new instance, otherwise `self` is modified. Defaults to `False`.

        Returns
        -------
        mat_ref: MatGF2
            The matrix in row-reduced echelon form.
        """
        ncols = self.data.shape[1] if ncols is None else ncols
        mat_ref = MatGF2(self.data) if copy else self

        return MatGF2(_elimination_jit(mat_ref.data, ncols=ncols, full_reduce=True))


def solve_f2_linear_system(mat: MatGF2, b: MatGF2) -> MatGF2:
    r"""Solve the linear system (LS) `mat @ x == b`.

    Parameters
    ----------
    mat : MatGF2
        Matrix with shape `(m, n)` containing the LS coefficients in row echelon form (REF).
    b : MatGF2
        Matrix with shape `(m,)` containing the constants column vector.

    Returns
    -------
    x : MatGF2
        Matrix with shape `(n,)` containing the solutions of the LS.

    Notes
    -----
    This function is not integrated in `:class: graphix.linalg.MatGF2` because it does not perform any checks on the form of `mat` to ensure that it is in REF or that the system is solvable.
    """
    return MatGF2(_solve_f2_linear_system_jit(mat.data, b.data))


@njit
def _solve_f2_linear_system_jit(
    mat_data: npt.NDArray[np.uint8], b_data: npt.NDArray[np.uint8]
) -> npt.NDArray[np.uint8]:
    """Wrap `:func:solve_f2_linear_system`. See docstring for details."""
    m, n = mat_data.shape
    x = np.zeros(n, dtype=np.uint8)

    # Find first row that is all-zero
    for i in range(m):
        for j in range(n):
            if mat_data[i, j] == 1:
                break  # Row is not zero → go to next row
        else:
            m_nonzero = i  # No break: this row is all-zero
            break
    else:
        m_nonzero = m

    # Backward substitution from row m_nonzero - 1 to 0
    for i in range(m_nonzero - 1, -1, -1):
        # Find first non-zero column in row i
        pivot = -1
        for j in range(n):
            if mat_data[i, j] == 1:
                pivot = j
                break

        # Sum x_k for k such that mat_data[i, k] == 1
        acc = 0
        for k in range(pivot, n):
            if mat_data[i, k] == 1:
                acc ^= x[k]

        x[pivot] = b_data[i] ^ acc

    return x


@njit
def _elimination_jit(mat_data: npt.NDArray[np.uint8], ncols: int, full_reduce: bool) -> npt.NDArray[np.uint8]:
    """Return row echelon form (REF) or row-reduced echelon form (RREF) by performing Gaussian elimination.

    Parameters
    ----------
    mat_data : npt.NDArray[np.uint8]
        Matrix to be gaussian-eliminated.
    n_cols : int
        Number of columns over which to perform Gaussian elimination.
    full_reduce : bool
        Flag determining the operation mode. Output is in RREF (respectively, REF) if `True` (repectively, `False`).

    Returns
    -------
    mat_data: npt.NDArray[np.uint8]
        The matrix in row(-reduced) echelon form.

    Notes
    -----
    Adapted from `:func: galois.FieldArray.row_reduce`, which renders the matrix in row-reduced echelon form (RREF) and specialized for GF(2)
    """
    m, n = mat_data.shape
    p = 0  # Pivot

    for j in range(ncols):
        # Find a pivot in column `j` at or below row `p`.
        for i in range(p, m):
            if mat_data[i, j] == 1:
                break  # `i` is a row with a pivot
        else:
            continue  # No break: column `j` does not have a pivot below row `p`.

        # Swap row `p` and `i`. The pivot is now located at row `p`.
        if i != p:
            for k in range(n):
                tmp = mat_data[i, k]
                mat_data[i, k] = mat_data[p, k]
                mat_data[p, k] = tmp

        if full_reduce:
            # Force zeros BELOW and ABOVE the pivot by xor-ing with the pivot row
            for k in range(m):
                if mat_data[k, j] == 1 and k != p:
                    for l in range(n):
                        mat_data[k, l] ^= mat_data[p, l]
        else:
            # Force zeros BELOW the pivot by xor-ing with the pivot row
            for k in range(p + 1, m):
                if mat_data[k, j] == 1:
                    for l in range(n):
                        mat_data[k, l] ^= mat_data[p, l]

        p += 1
        if p == m:
            break

    return mat_data
