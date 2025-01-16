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
    session.run("pytest")
