"""Run tests with nox."""

from __future__ import annotations

from tempfile import TemporaryDirectory

import nox
from nox import Session


@nox.session(python=["3.9", "3.10", "3.11", "3.12", "3.13"])
def tests_minimal(session: Session) -> None:
    """Run the test suite with minimal dependencies."""
    session.install("-e", ".")
    session.install("pytest", "pytest-mock", "psutil")
    session.run("pytest")


@nox.session(python=["3.9", "3.10", "3.11", "3.12", "3.13"])
def tests(session: Session) -> None:
    """Run the test suite with full dependencies."""
    session.install("-e", ".[dev,extra]")
    session.run("pytest", "--doctest-modules")


@nox.session(python=["3.9", "3.10", "3.11", "3.12", "3.13"])
def tests_symbolic(session: Session) -> None:
    """Run the test suite of graphix-symbolic."""
    session.install("-e", ".[dev]")
    # Use `session.cd` as a context manager to ensure that the
    # working directory is restored afterward. This is important
    # because Windows cannot delete a temporary directory while it
    # is the working directory.
    with TemporaryDirectory() as tmpdir, session.cd(tmpdir):
        # See https://github.com/TeamGraphix/graphix-symbolic/pull/3
        # If you need a specific branch:
        session.run("git", "clone", "-b", "implement_abs", "https://github.com/thierry-martinez/graphix-symbolic")
        # session.run("git", "clone", "https://github.com/TeamGraphix/graphix-symbolic")
        with session.cd("graphix-symbolic"):
            session.run("pytest")
