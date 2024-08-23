"""Utilities."""

from __future__ import annotations

from typing import Any, Never

import annotated_types
import typing_extensions

if typing_extensions.TYPE_CHECKING:
    from collections.abc import Iterable

PositiveOrNullInt = typing_extensions.Annotated[int, annotated_types.Ge(0)]  # includes 0


def check_list_elements(l: Iterable[Any], ty: type) -> None:
    for index, item in enumerate(l):
        if not isinstance(item, ty):
            raise TypeError(f"data[{index}] has type {type(item)} whereas {ty} is expected")


def disable_init(cls: type) -> type:
    class _Inner(cls):
        def __init__(self, *_args: Any, **_kwargs: Any) -> Never:
            raise RuntimeError("This class should not be instantiated.")

    return _Inner
