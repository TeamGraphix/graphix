"""Simulation backends."""

from __future__ import annotations

from typing import Literal

from graphix.sim.base_backend import Backend
from graphix.sim.data import Data
from graphix.sim.density_matrix import DensityMatrix, DensityMatrixBackend
from graphix.sim.statevec import Statevec, StatevectorBackend
from graphix.sim.tensornet import MBQCTensorNet, TensorNetworkBackend

_BuiltinBackendState = DensityMatrix | Statevec | MBQCTensorNet
_BuiltinBackend = DensityMatrixBackend | StatevectorBackend | TensorNetworkBackend
_BackendLiteral = Literal["statevector", "densitymatrix", "tensornetwork", "mps"]

__all__ = [
    "Backend",
    "Data",
    "DensityMatrix",
    "DensityMatrixBackend",
    "Statevec",
    "StatevectorBackend",
    "_BackendLiteral",
    "_BuiltinBackend",
    "_BuiltinBackendState",
]
