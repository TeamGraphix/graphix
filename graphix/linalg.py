import galois
import numpy as np
import sympy as sp


class MatGF2:
    """Matrix on GF2 field"""

    def __init__(self, data):
        """constructor for matrix of GF2

        Parameters
        ----------
        data: array_like
            input data
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

    def copy(self):
        return MatGF2(self.data.copy())

    def add_row(self, array_to_add=None, row=None):
        """add a row to the matrix

        Parameters
        ----------
        array_to_add: array_like(optional)
            row to add. Defaults to None. if None, add a zero row.
        row: int(optional)
            index to add a new row. Defaults to None.
        """
        if row is None:
            row = self.data.shape[0]
        if array_to_add is None:
            array_to_add = np.zeros((1, self.data.shape[1]), dtype=int)
        array_to_add = array_to_add.reshape((1, self.data.shape[1]))
        self.data = np.insert(self.data, row, array_to_add, axis=0)

    def add_col(self, array_to_add=None, col=None):
        """add a column to the matrix

        Parameters
        ----------
        array_to_add: array_like(optional)
            column to add. Defaults to None. if None, add a zero column.
        col: int(optional)
            index to add a new column. Defaults to None.
        """
        if col is None:
            col = self.data.shape[1]
        if array_to_add is None:
            array_to_add = np.zeros((1, self.data.shape[0]), dtype=int)
        array_to_add = array_to_add.reshape((1, self.data.shape[0]))
        self.data = np.insert(self.data, col, array_to_add, axis=1)

    def remove_row(self, row):
        """remove a row from the matrix

        Parameters
        ----------
        row: int
            index to remove a row
        """
        self.data = np.delete(self.data, row, axis=0)

    def remove_col(self, col):
        """remove a column from the matrix

        Parameters
        ----------
        col: int
            index to remove a column
        """
        self.data = np.delete(self.data, col, axis=1)

    def swap_row(self, row1, row2):
        """swap two rows

        Parameters
        ----------
        row1: int
            row index
        row2: int
            row index
        """
        self.data[[row1, row2]] = self.data[[row2, row1]]

    def swap_col(self, col1, col2):
        """swap two columns

        Parameters
        ----------
        col1: int
            column index
        col2: int
            column index
        """
        self.data[:, [col1, col2]] = self.data[:, [col2, col1]]

    def is_canonical_form(self):
        """check if the matrix is in a canonical(Row reduced echelon form) form

        Returns
        -------
        bool: bool
            True if the matrix is in canonical form
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

        Returns
        -------
        int: int
            rank of the matrix
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
        where X is an arbitrary matrix

        Parameters
        ----------
        b: array_like(otional)
            Left hand side of the system of equations. Defaults to None.
        copy: bool(optional)
            copy the matrix or not. Defaults to False.

        Returns
        -------
        A: MatGF2
            forward eliminated matrix
        b: MatGF2
            forward eliminated right hand side
        row_pertumutation: list
            row permutation
        col_pertumutation: list
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

        Parameters
        ----------
        b: array_like
            right hand side of the system of equations

        Returns
        -------
        x: sympy.MutableDenseMatrix
            answer of the system of equations
        kernels: list-of-sympy.Symbol
            kernel of the matrix.
            matrix x contains sympy.Symbol if the input matrix is not full rank.
            nan-column vector means that there is no solution.
        """
        rank = self.get_rank()
        b = MatGF2(b)
        x = list()
        kernels = sp.symbols("x0:%d" % (self.data.shape[1] - rank))
        for col in range(b.data.shape[1]):
            x_col = list()
            b_col = b.data[:, col]
            if np.count_nonzero(b_col[rank:]) != 0:
                x_col = [sp.nan for i in range(rank)]
                x.append(x_col)
                continue
            for row in range(rank - 1, -1, -1):
                sol = sp.true if b_col[row] == 1 else sp.false
                kernel_index = np.nonzero(self.data[row, rank:])[0]
                for k in kernel_index:
                    sol ^= kernels[k]
                x_col.insert(0, sol)
            x.append(x_col)

        x = np.array(x).T

        return x, kernels
