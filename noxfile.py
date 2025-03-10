"""Run tests with nox."""

from __future__ import annotations

import nox
from nox import Session


@nox.session(python=["3.8", "3.9", "3.10", "3.11", "3.12"])
def tests_minimal(session: Session) -> None:
    """Run the test suite with minimal dependencies."""
    session.install("-e", ".")
    session.install("pytest", "pytest-mock", "psutil")
    session.run("pytest")


@nox.session(python=["3.8", "3.9", "3.10", "3.11", "3.12"])
def tests(session: Session) -> None:
    """Run the test suite with full dependencies."""
    session.install("-e", ".[dev]")
    session.run("pytest", "--doctest-modules")


@nox.session(python=["3.8", "3.9", "3.10", "3.11", "3.12"])
def tests_symbolic(session: Session) -> None:
    """Run the test suite of graphix-symbolic."""
    session.install("-e", ".[dev]")
    ## If you need a specific branch:
    # session.run("git", "clone", "-b", "branch-name", "https://github.com/TeamGraphix/graphix-symbolic")
    session.run("git", "clone", "https://github.com/TeamGraphix/graphix-symbolic")
    session.cd("graphix-symbolic")
    session.run("pytest")
