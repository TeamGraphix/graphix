import galois
import numpy as np
from sympy import Expr, Integer, MutableDenseMatrix, Xor, nan, symbols


class MatGF2:
    """Matrix on GF2 field"""

    def __init__(self, data):
        """constructor for matrix of GF2

        Args:
            data (array_like): input data
        """
        if isinstance(data, MatGF2):
            self.data = data.data
        else:
            self.data = galois.GF2(data)

    def __repr__(self):
        return str(self.data)

    def __str__(self):
        return str(self.data)

    def __eq__(self, other):
        return np.all(self.data == other.data)

    def __add__(self, other):
        if isinstance(other, np.ndarray):
            other = MatGF2(other)
        return MatGF2(self.data + other.data)

    def __sub__(self, other):
        if isinstance(other, np.ndarray):
            other = MatGF2(other)
        return MatGF2(self.data - other.data)

    def add_row(self, row=None):
        """add a row to the matrix

        Args:
            row (int, optional): index to add a new row. Defaults to None.
        """
        if row is None:
            row = self.data.shape[0]
        self.data = np.insert(self.data, row, 0, axis=0)

    def add_col(self, col=None):
        """add a column to the matrix

        Args:
            col (int, optional): index to add a new column. Defaults to None.
        """
        if col is None:
            col = self.data.shape[1]
        self.data = np.insert(self.data, col, 0, axis=1)

    def remove_row(self, row):
        """remove a row from the matrix

        Args:
            row (int): index to remove a row
        """
        self.data = np.delete(self.data, row, axis=0)

    def remove_col(self, col):
        """remove a column from the matrix

        Args:
            col (int): index to remove a column
        """
        self.data = np.delete(self.data, col, axis=1)

    def swap_row(self, row1, row2):
        """swap two rows

        Args:
            row1 (int): row index
            row2 (int): row index
        """
        self.data[[row1, row2]] = self.data[[row2, row1]]

    def swap_col(self, col1, col2):
        """swap two columns

        Args:
            col1 (int): column index
            col2 (int): column index
        """
        self.data[:, [col1, col2]] = self.data[:, [col2, col1]]

    def is_canonical_form(self):
        """check if the matrix is in a canonical(Row reduced echelon form) form

        Returns:
            bool: True if the matrix is in canonical form
        """
        diag = self.data.diagonal()
        nonzero_diag_index = diag.nonzero()[0]

        rank = len(nonzero_diag_index)
        for i in range(len(nonzero_diag_index)):
            if diag[nonzero_diag_index[i]] == 0:
                if np.count_nonzero(diag[i:]) != 0:
                    break
                else:
                    return False

        ref_array = MatGF2(np.diag(np.diagonal(self.data[:rank, :rank])))
        if np.count_nonzero(self.data[:rank, :rank] - ref_array.data) != 0:
            return False

        if np.count_nonzero(self.data[rank:, :]) != 0:
            return False

        return True

    def get_rank(self):
        """get the rank of the matrix

        Returns:
            int: rank of the matrix
        """
        if not self.is_canonical_form():
            A = self.forward_eliminate(copy=True)[0]
        else:
            A = self
        nonzero_index = np.diag(A.data).nonzero()
        return len(nonzero_index[0])

    def forward_eliminate(self, b=None, copy=False):
        """forward eliminate the matrix

        |A B| --\ |I X|
        |C D| --/ |0 0|

        Args:
            b (array_like, otional): Left hand side of the system of equations. Defaults to None.
            copy (bool, optional): copy the matrix or not. Defaults to False.

        Returns:
            (MatGF2, MatGF2, list list): the forward eliminated matrix,
            the forward eliminated right hand side,
            row permutation,
            column permutation
        """
        if copy:
            A = MatGF2(self.data)
        else:
            A = self
        if b is None:
            b = np.zeros((A.data.shape[0], 1), dtype=int)
        b = MatGF2(b)
        # Remember the row and column order
        row_pertumutation = [i for i in range(A.data.shape[0])]
        col_pertumutation = [i for i in range(A.data.shape[1])]

        # Gauss-Jordan Elimination
        max_rank = min(A.data.shape)
        for row in range(max_rank):
            if A.data[row, row] == 0:
                pivot = A.data[row:, row:].nonzero()
                if len(pivot[0]) == 0:
                    break
                pivot_row = pivot[0][0] + row
                if pivot_row != row:
                    A.swap_row(row, pivot_row)
                    b.swap_row(row, pivot_row)
                    row_pertumutation[row] = pivot_row
                    row_pertumutation[pivot_row] = row
                pivot_col = pivot[1][0] + row
                if pivot_col != row:
                    A.swap_col(row, pivot_col)
                    col_pertumutation[row] = pivot_col
                    col_pertumutation[pivot_col] = row
            eliminate_rows = set(A.data[:, row].nonzero()[0]) - {row}
            for eliminate_row in eliminate_rows:
                A.data[eliminate_row] += A.data[row]
                b.data[eliminate_row] += b.data[row]
        return A, b, row_pertumutation, col_pertumutation

    def backward_substitute(self, b):
        """backward substitute the matrix

        Args:
            b (array_like): right hand side of the system of equations

        Returns:
            (sympy.MutableDenseMatrix, list-of-sympy.Symbol): answer of the system of equations, kernel of the matrix.
            matrix x contains sympy.Symbol if the input matrix is not full rank.
            nan-column vector means that there is no solution.
        """
        rank = self.get_rank()
        b = MatGF2(b)
        x = MutableDenseMatrix(np.zeros((self.data.shape[1], b.data.shape[1])))
        kernels = symbols("x0:%d" % (self.data.shape[1] - rank))
        for col in range(b.data.shape[1]):
            b_col = b.data[:, col]
            if np.count_nonzero(b_col[rank:]) != 0:
                for row in range(self.data.shape[1]):
                    x[row, col] = nan
            for row in range(rank - 1, -1, -1):
                x[row, col] = Integer(b_col[row])
                kernel_index = np.nonzero(self.data[row, rank:])[0]
                for k in kernel_index:
                    x[row, col] = Expr(Xor(x[row, col], kernels[k]))

        return x, kernels
