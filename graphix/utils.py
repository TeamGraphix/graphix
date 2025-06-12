"""Utilities."""

from __future__ import annotations

import dataclasses
import sys
import typing
from dataclasses import MISSING
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Literal, SupportsInt, TypeVar

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    # these live only in the stub package, not at runtime
    from _typeshed import DataclassInstance

_T = TypeVar("_T")


def check_list_elements(l: Iterable[_T], ty: type[_T]) -> None:
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

    import inspect  # noqa: PLC0415

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


def is_integer(value: SupportsInt) -> bool:
    """Return `True` if `value` is an integer, `False` otherwise."""
    return value == int(value)


G = TypeVar("G", bound=np.generic)


@typing.overload
def lock(data: npt.NDArray[Any]) -> npt.NDArray[np.complex128]: ...


@typing.overload
def lock(data: npt.NDArray[Any], dtype: type[G]) -> npt.NDArray[G]: ...


def lock(data: npt.NDArray[Any], dtype: type = np.complex128) -> npt.NDArray[Any]:
    """Create a true immutable view.

    data must not have aliasing references, otherwise users can still turn on writeable flag of m.
    """
    m: npt.NDArray[Any] = data.astype(dtype)
    m.flags.writeable = False
    v = m.view()
    assert not v.flags.writeable
    return v


def iter_empty(it: Iterator[_T]) -> bool:
    """Check if an iterable is empty.

    Notes
    -----
    This function consumes the iterator.
    """
    return all(False for _ in it)


class DataclassPrettyPrintMixin:
    """
    Mixin for a concise, eval-friendly `repr` of dataclasses.

    Compared to the default dataclass `repr`:
      - Class variables are omitted (dataclasses.fields only returns actual fields).
      - Fields whose values equal their defaults are omitted.
      - Field names are only shown when preceding fields have been omitted, ensuring positional listings when possible.

    Use with `@dataclass(repr=False)` on the target class.
    """

    def __repr__(self: DataclassInstance) -> str:
        """Return a representation string for a dataclass."""
        cls_name = type(self).__name__
        arguments = []
        saw_omitted = False
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if field.default is not MISSING or field.default_factory is not MISSING:
                default = field.default_factory() if field.default_factory is not MISSING else field.default
                if value == default:
                    saw_omitted = True
                    continue
            custom_repr = field.metadata.get("repr")
            value_str = custom_repr(value) if custom_repr else repr(value)
            if saw_omitted:
                arguments.append(f"{field.name}={value_str}")
            else:
                arguments.append(value_str)
        arguments_str = ", ".join(arguments)
        return f"{cls_name}({arguments_str})"


class EnumPrettyPrintMixin:
    """
    Mixin to provide a concise, eval-friendly repr for Enum members.

    Compared to the default `<ClassName.MEMBER_NAME: value>`, this mixin's `__repr__`
    returns `ClassName.MEMBER_NAME`, which can be evaluated in Python (assuming the
    enum class is in scope) to retrieve the same member.
    """

    def __repr__(self) -> str:
        """
        Return a representation string of an Enum member.

        Returns
        -------
        str
            A string in the form `ClassName.MEMBER_NAME`.
        """
        # Equivalently (as of Python 3.12), `str(value)` also produces
        # "ClassName.MEMBER_NAME", but we build it explicitly here for
        # clarity.
        if not isinstance(self, Enum):
            msg = "EnumMixin can only be used with Enum classes."
            raise TypeError(msg)
        return f"{self.__class__.__name__}.{self.name}"
