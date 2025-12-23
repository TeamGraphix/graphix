"""Common utils for airspeed velocity benchmarking."""

from __future__ import annotations

from numpy.random import PCG64, Generator

SEED = 25


def fx_rng() -> Generator:
    """Return a seeded random generator."""
    return Generator(PCG64(SEED))
