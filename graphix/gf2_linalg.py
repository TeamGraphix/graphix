"""Modulo 2 linear equations."""

from __future__ import annotations

import numba
import numpy as np
import numpy.typing as npt


@numba.njit(cache=True)
def _move_pivot(work: npt.NDArray[np.bool_], perm: npt.NDArray[np.int64], i: int) -> bool:
    """Move the first non-zero element to (i, i) and update permutation."""
    rows, width = work.shape
    cols = perm.size
    for r in range(i, rows):
        for c in range(i, cols):
            if not work[r, c]:
                continue
            if i != r:
                for x in range(width):
                    work[i, x], work[r, x] = work[r, x], work[i, x]
            if i != c:
                for x in range(rows):
                    work[x, i], work[x, c] = work[x, c], work[x, i]
                perm[i], perm[c] = perm[c], perm[i]
            return True
    return False


@numba.njit(cache=True)
def _eliminate_lower_impl(work: npt.NDArray[np.bool_], i: int) -> None:
    """Eliminate the lower part using (i, i) as pivot."""
    rows, _ = work.shape
    src = work[i, i:]
    for r in range(i + 1, rows):
        if work[r, i]:
            work[r, i:] ^= src


@numba.njit(cache=True)
def _eliminate_lower(work: npt.NDArray[np.bool_], perm: npt.NDArray[np.int64]) -> int:
    """Eliminate the lower part."""
    rows, _ = work.shape
    cols = perm.size
    rmax = min(rows, cols)
    for r in range(rmax):
        if not _move_pivot(work, perm, r):
            return r
        _eliminate_lower_impl(work, r)
    return rmax


@numba.njit(cache=True)
def _eliminate_upper_impl(work: npt.NDArray[np.bool_], i: int) -> None:
    """Eliminate the upper part using (i, i) as pivot."""
    src = work[i, i:]
    for r in range(i):
        if work[r, i]:
            work[r, i:] ^= src


@numba.njit(cache=True)
def _eliminate_upper(work: npt.NDArray[np.bool_], rank: int) -> None:
    """Eliminate the upper part."""
    for r in range(rank - 1, -1, -1):
        _eliminate_upper_impl(work, r)


class GF2Solver:
    """Solve modulo 2 linear equations."""

    __work: npt.NDArray[np.bool_]
    __perm: npt.NDArray[np.int64]
    __rank: int | None

    def __init__(self, lhs: npt.NDArray[np.bool_], rhs: npt.NDArray[np.bool_]) -> None:
        """Initialize the solver.

        Args
        ----
        lhs: `npt.NDArray[np.bool_]`
            The left-hand side matrix.
        rhs: `npt.NDArray[np.bool_]`
            The right-hand side matrix.
            Can handle multiple right-hand sides at once.

        Raises
        ------
        ValueError
            On inconsistent shapes.

        TypeError
            On invalid types.

        """
        rows, cols = lhs.shape
        _, neqs = rhs.shape
        if rhs.shape != (rows, neqs):
            msg = "lhs and rhs must have the same number of rows."
            raise ValueError(msg)
        if rhs.dtype != np.bool_:
            msg = "rhs must be a boolean matrix."
            raise TypeError(msg)
        if lhs.dtype != np.bool_:
            msg = "lhs must be a boolean matrix."
            raise TypeError(msg)
        self.__work = np.hstack([lhs, rhs], dtype=np.bool_)
        self.__perm = np.arange(cols, dtype=np.int64)
        self.__rank = None

    def _eliminate_lower(self) -> None:
        self.__rank = _eliminate_lower(self.__work, self.__perm)

    def _eliminate_upper(self) -> None:
        if self.__rank is None:
            msg = "_eliminate_lower must be called before eliminate_upper."
            raise RuntimeError(msg)
        _eliminate_upper(self.__work, self.__rank)

    def _eliminate(self) -> None:
        self._eliminate_lower()
        self._eliminate_upper()

    @property
    def work(self) -> npt.NDArray[np.bool_]:
        """Get the working storage."""
        return self.__work.copy()

    @property
    def perm(self) -> npt.NDArray[np.int64]:
        """Get the permutation."""
        return self.__perm.copy()

    @property
    def rank(self) -> int | None:
        """Get the rank.

        Result is `None` if rank is not yet computed.
        """
        return self.__rank

    def solve(self, ieq: int) -> npt.NDArray[np.bool_] | None:
        """Solve the i-th equation.

        Args
        ----
        ieq: `int`
            The equation index.

        Raises
        ------
        ValueError
            On invalid equation index.

        """
        _, width = self.__work.shape
        cols = self.__perm.size
        neqs = width - cols
        if not (0 <= ieq < neqs):
            msg = "ieq out of bounds."
            raise ValueError(msg)
        if self.__rank is None:
            self._eliminate()
        assert self.__rank is not None
        target = self.__work[:, cols + ieq]
        if any(target[self.__rank :]):
            return None
        ind = np.arange(self.__rank, dtype=np.int64)
        ret = np.zeros(cols, dtype=np.bool_)
        ret[self.__perm[ind]] = target[ind]
        return ret

    def _check(self) -> None:
        """Perform debug check.

        Raises
        ------
        RuntimeError
            On any error.

        """
        rows, _ = self.__work.shape
        cols = self.__perm.size
        if self.__rank is None:
            msg = "rank is not yet computed."
            raise RuntimeError(msg)
        for i in range(self.__rank):
            if not self.__work[i, i]:
                msg = "Inconsistent rank."
                raise RuntimeError(msg)
            if any(self.__work[i, :i]):
                msg = "Lower part is not yet eliminated."
                raise RuntimeError(msg)
            if any(self.__work[i, i + 1 : self.__rank]):
                msg = "Upper part is not yet eliminated."
                raise RuntimeError(msg)
        for i in range(self.__rank, rows):
            if any(self.__work[i, :cols]):
                msg = "Inconsistent rank."
                raise RuntimeError(msg)
