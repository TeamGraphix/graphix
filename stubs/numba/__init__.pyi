from typing import Any, Callable, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])

def njit(f: _F) -> _F: ...
