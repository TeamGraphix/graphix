"""Run tests with nox."""

from __future__ import annotations

import nox
from nox import Session

TESTING_LIBS = ["psutil", "pytest", "pytest-mock", "pytest-sugar"]


@nox.session(python=["3.8", "3.9", "3.10", "3.11", "3.12"])
def runtest_mini(session: Session) -> None:
    """Run the test suite with minimal dependencies."""
    session.install("-e", ".")
    session.install(*TESTING_LIBS)
    session.run("pytest")


@nox.session(python=["3.8", "3.9", "3.10", "3.11", "3.12"])
def runtest_full(session: Session) -> None:
    """Run the test suite with full dependencies."""
    session.install("-e", ".[dev]")
    session.install(*TESTING_LIBS)
    session.run("pytest")


@nox.session()
def runtest_cov(session: Session) -> None:
    """Measure test coverage."""
    session.install("-e", ".[dev]")
    session.install(*TESTING_LIBS, "pytest-cov")
    session.run("pytest", "--cov=./graphix", "--cov-report=xml", "--cov-report=term")
