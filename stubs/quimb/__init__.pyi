from collections.abc import Iterator
from typing import Generic, TypeVar

_T = TypeVar("_T")

class oset(Generic[_T]):  # noqa: N801
    def __iter__(self) -> Iterator[_T]: ...
    def popleft(self) -> _T: ...
