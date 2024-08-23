"""Utilities."""

from __future__ import annotations

from typing import Any

import annotated_types
import typing_extensions

if typing_extensions.TYPE_CHECKING:
    from collections.abc import Iterable

PositiveOrNullInt = typing_extensions.Annotated[int, annotated_types.Ge(0)]  # includes 0


def check_list_elements(l: Iterable[Any], ty: type) -> None:
    for index, item in enumerate(l):
        if not isinstance(item, ty):
            raise TypeError(f"data[{index}] has type {type(item)} whereas {ty} is expected")
