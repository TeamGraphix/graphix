r"""Performant module for linear algebra on :math:`\mathbb F_2` field."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numba as nb
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from typing import Self


class MatGF2(npt.NDArray[np.uint8]):
    r"""Custom implementation of :math:`\mathbb F_2` matrices. This class specializes `:class:np.ndarray` to the :math:`\mathbb F_2` field with increased efficiency."""

    def __new__(cls, data: npt.ArrayLike) -> Self:
        """Instantiate new `MatGF2` object.

        Parameters
        ----------
        data : array
            Data in array
        dtype : npt.DTypeLike
            Optional, defaults to `np.uint8`.

        Return
        -------
            MatGF2
        """
        arr = np.array(data, dtype=np.uint8)
        return super().__new__(cls, shape=arr.shape, dtype=arr.dtype, buffer=arr)

    def mat_mul(self, other: MatGF2 | npt.NDArray[np.uint8]) -> MatGF2:
        r"""Multiply two matrices.

        Parameters
        ----------
        other : array
            Matrix that right-multiplies `self`.

        Returns
        -------
        MatGF2
            Matrix product `self` @  `other` in :math:`\mathbb F_2`.

        Notes
        -----
        This function is a wrapper over :func:`_mat_mul_jit` which is a just-time compiled implementation of the matrix multiplication in :math:`\mathbb F_2`. It is more efficient than `galois.GF2.__matmul__` when the matrix `self` is sparse.
        The implementation assumes that the arguments have the right dimensions.
        """
        return MatGF2(_mat_mul_jit(self, other))

    def compute_rank(self) -> int:
        """Get the rank of the matrix.

        Returns
        -------
        int : int
            Rank of the matrix.
        """
        mat_a = self.row_reduction()
        return int(np.sum(mat_a.any(axis=1)))

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
        m, n = self.shape
        if m > n:
            return None

        ident = np.eye(m, dtype=np.uint8)
        aug = np.hstack([self.data, ident]).view(MatGF2)
        red = aug.row_reduction(ncols=n)  # Reduced row echelon form

        # Check that rank of right block is equal to the number of rows.
        # We don't use `MatGF2.compute_rank()` to avoid row-reducing twice.
        if m != np.count_nonzero(red[:, :n].any(axis=1)):
            return None
        rinv = np.zeros((n, m), dtype=np.uint8).view(MatGF2)

        for i, row in enumerate(red):
            j = np.flatnonzero(row)[0]  # Column index corresponding to the leading 1 in row `i`.
            rinv[j, :] = red[i, n:]

        return rinv

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
        m, n = self.shape

        ident = np.eye(n, dtype=np.uint8)
        ref = np.hstack([self.T, ident]).view(MatGF2)
        ref.gauss_elimination(ncols=m)
        row_idxs = np.flatnonzero(~ref[:, :m].any(axis=1))  # Row indices of the 0-rows in the first block of `ref`.

        return ref[row_idxs, m:].view(MatGF2)

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
        ncols = self.shape[1] if ncols is None else ncols
        mat_ref = MatGF2(self) if copy else self

        return MatGF2(_elimination_jit(mat_ref, ncols=ncols, full_reduce=False))

    def row_reduction(self, ncols: int | None = None, copy: bool = False) -> MatGF2:
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
        ncols = self.shape[1] if ncols is None else ncols
        mat_ref = self.copy() if copy else self

        return MatGF2(_elimination_jit(mat_ref, ncols=ncols, full_reduce=True))


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
    return MatGF2(_solve_f2_linear_system_jit(mat, b))


@nb.njit("uint8[::1](uint8[:,::1], uint8[::1])")
def _solve_f2_linear_system_jit(
    mat_data: npt.NDArray[np.uint8], b_data: npt.NDArray[np.uint8]
) -> npt.NDArray[np.uint8]:
    """See docstring of `:func:solve_f2_linear_system` for details."""
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


@nb.njit("uint8[:,::1](uint8[:,::1], uint64, boolean)")
def _elimination_jit(mat_data: npt.NDArray[np.uint8], ncols: int, full_reduce: bool) -> npt.NDArray[np.uint8]:
    r"""Return row echelon form (REF) or row-reduced echelon form (RREF) by performing Gaussian elimination.

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
    Adapted from `:func: galois.FieldArray.row_reduction`, which renders the matrix in row-reduced echelon form (RREF) and specialized for :math:`\mathbb F_2`.

    Row echelon form (REF):
        1. All rows having only zero entries are at the bottom.
        2. The leading entry of every nonzero row is on the right of the leading entry of every row above.
        3. (1) and (2) imply that all entries in a column below a leading coefficient are zeros.
        4. It's the result of Gaussian elimination.

    For matrices over :math:`\mathbb F_2` the only difference between REF and RREF is that elements above a leading 1 can be non-zero in REF but must be 0 in RREF.
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
                mat_data[i, k], mat_data[p, k] = mat_data[p, k], mat_data[i, k]

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


@nb.njit("uint8[:,::1](uint8[:,::1], uint8[:,::1])", parallel=True)
def _mat_mul_jit(m1: npt.NDArray[np.uint8], m2: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """See docstring of `:func:MatGF2.__matmul__` for details."""
    m, l = m1.shape
    _, n = m2.shape

    res = np.zeros((m, n), dtype=np.uint8)

    for i in nb.prange(m):
        for k in nb.prange(l):
            if m1[i, k] == 1:
                for j in range(n):
                    res[i, j] = np.bitwise_xor(res[i, j], m2[k, j])

    return res
