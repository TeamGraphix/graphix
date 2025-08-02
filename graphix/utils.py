"""Utilities."""

from __future__ import annotations

import inspect
import sys
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, SupportsInt, TypeVar, cast

import numpy as np
import numpy.typing as npt
import typing_extensions

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

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

    # Type annotation to work around a regression in mypy 1.17, see https://github.com/python/mypy/issues/19458
    ann: Any | None = inspect.get_annotations(cls, eval_str=True, locals=scope).get("kind")
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


_ValueT = TypeVar("_ValueT")


class Validator(ABC, Generic[_ValueT]):
    """Descriptor to validate value.

    https://docs.python.org/3/howto/descriptor.html#custom-validators
    """

    def __set_name__(self, owner: object, name: str) -> None:
        """Set private field name."""
        self.private_name = "_" + name

    def __get__(self, obj: object, objtype: object = None) -> _ValueT:
        """Get the validated value from the private field."""
        return cast("_ValueT", getattr(obj, self.private_name))

    def __set__(self, obj: object, value: _ValueT) -> None:
        """Validate and set the value in the private field."""
        self.validate(value)
        setattr(obj, self.private_name, value)

    @abstractmethod
    def validate(self, value: _ValueT) -> None:
        """Validate the assigned value."""


@dataclass
class Number(Validator[float]):
    """Descriptor to validate numbers with given bounds.

    https://docs.python.org/3/howto/descriptor.html#custom-validators
    """

    minvalue: float | None = None
    maxvalue: float | None = None

    @typing_extensions.override
    def validate(self, value: object) -> None:
        """Validate the assigned value."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"Expected {value!r} to be an int or float")
        if self.minvalue is not None and value < self.minvalue:
            raise ValueError(f"Expected {value!r} to be at least {self.minvalue!r}")
        if self.maxvalue is not None and value > self.maxvalue:
            raise ValueError(f"Expected {value!r} to be no more than {self.maxvalue!r}")


class Probability(Number):
    """Descriptor for probability (between 0 and 1)."""

    def __init__(self) -> None:
        super().__init__(minvalue=0, maxvalue=1)
