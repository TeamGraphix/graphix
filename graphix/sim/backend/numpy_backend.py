from __future__ import annotations

import numpy as np

from .abstract_backend import AbstractBackend


class NumPyBackend(AbstractBackend):
    """A backend that uses NumPy for its computations."""

    def sin(self, x: np.ndarray):
        return np.sin(x)

    def cos(self, x: np.ndarray):
        return np.cos(x)

    def sum(self, x: np.ndarray):
        return np.sum(x)
