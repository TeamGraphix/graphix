from collections.abc import Iterator, Mapping, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
from cotengra.oe import PathOptimizer
from quimb import oset
from typing_extensions import Self

class Tensor:
    data: npt.NDArray[np.generic]

    def __init__(
        self,
        data: float | Tensor | npt.NDArray[np.generic] = 1.0,
        inds: Sequence[str] = (),
        tags: Sequence[str] | None = None,
        left_inds: Sequence[str] | None = None,
    ) -> None: ...
    def reindex(self, retag_map: Mapping[str, str], inplace: bool = False) -> Tensor: ...
    def retag(self, retag_map: Mapping[str, str], inplace: bool = False) -> Tensor: ...
    def split(
        self,
        left_inds: str | Sequence[str],
        max_bond: int | None = None,
        bond_ind: str | None = None,
        right_inds: str | Sequence[str] | None = None,
    ) -> TensorNetwork: ...
    @property
    def H(self) -> Tensor: ...  # noqa: N802

class TensorNetwork:
    tensor_map: dict[str, Tensor]
    tag_map: dict[str, str]

    def __init__(
        self,
        ts: Sequence[Tensor | TensorNetwork] | TensorNetwork = (),
        *,
        virtual: bool = False,
        check_collisions: bool = True,
    ) -> None: ...
    def _get_tids_from_tags(
        self, tags: Sequence[str] | str | int | None, which: Literal["all", "any", "!all", "!any"] = "all"
    ) -> oset[str]: ...
    def _get_tids_from_inds(
        self, inds: Sequence[str] | str | int | None, which: Literal["all", "any", "!all", "!any"] = "all"
    ) -> oset[str]: ...
    def __iter__(self) -> Iterator[Tensor]: ...
    def add(
        self,
        t: Tensor | TensorNetwork | Sequence[Tensor | TensorNetwork],
        virtual: bool = False,
        check_collisions: bool = True,
    ) -> None: ...
    def add_tensor(self, tensor: Tensor, tid: str | None = None, virtual: bool = False) -> None: ...
    def add_tensor_network(self, tn: TensorNetwork, virtual: bool = False, check_collisions: bool = False) -> None: ...
    def conj(self) -> TensorNetwork: ...
    def contract(
        self,
        *others: Sequence[TensorNetwork],
        output_inds: Sequence[str] | None = None,
        optimize: str | PathOptimizer | None = None,
    ) -> float: ...
    def copy(self, virtual: bool = False, deep: bool = False) -> TensorNetwork: ...
    def full_simplify(
        self,
        seq: str = "ADCR",
        output_inds: Sequence[str] | None = None,
        atol: float = 1e-12,
        equalize_norms: bool = False,
        inplace: bool = False,
        progbar: bool = False,
    ) -> Self: ...
    def split(
        self,
        left_inds: str | Sequence[str],
        max_bond: int | None = None,
        bond_ind: str | None = None,
        right_inds: str | Sequence[str] | None = None,
    ) -> TensorNetwork: ...
    @property
    def tensors(self) -> tuple[Tensor, ...]: ...

def rand_uuid(base: str = "") -> str: ...
