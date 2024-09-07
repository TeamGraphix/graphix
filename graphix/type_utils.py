"""Type utilities."""

from __future__ import annotations

import sys
import typing
from typing import Any, ClassVar, Literal


def check_list_elements(l, ty):
    """Check that every element of the list has the given type."""
    for index, item in enumerate(l):
        if not isinstance(item, ty):
            raise TypeError(f"data[{index}] has type {type(item)} whereas {ty} is expected")


def check_kind(cls: type, scope: dict[str, Any]) -> None:
    """Check that the class has a kind attribute."""
    if not hasattr(cls, "kind"):
        msg = f"{cls.__name__} must have a tag attribute named kind."
        raise TypeError(msg)
    if sys.version_info < (3, 10):
        # MEMO: `inspect.get_annotations` unavailable
        return

    import inspect

    ann = inspect.get_annotations(cls, eval_str=True, locals=scope).get("kind")
    if ann is None:
        msg = "kind must be annotated."
        raise TypeError(msg)
    if typing.get_origin(ann) is not ClassVar:
        msg = "Tag attribute must be a class variable."
        raise TypeError(msg)
    (ann,) = typing.get_args(ann)
    if typing.get_origin(ann) is not Literal:
        msg = "Tag attribute must be a literal."
        raise TypeError(msg)
