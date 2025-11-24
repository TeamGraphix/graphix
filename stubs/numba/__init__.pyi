from collections.abc import Callable
from typing import Any, TypeVar, overload

_F = TypeVar("_F", bound=Callable[..., Any])

@overload
def njit(f: _F) -> _F: ...
@overload
def njit(ty: str, parallel: bool = False) -> Callable[[_F], _F]: ...
def prange(low: int, high: int | None = None) -> range: ...
