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


# TODO: Add 3.13 CI
@nox.session(python=["3.9", "3.10", "3.11", "3.12"])
def tests_symbolic(session: Session) -> None:
    """Run the test suite of graphix-symbolic."""
    session.install("-e", ".[dev]")
    # Temporary directory, otherwise nox clones graphix-symbolic in the working directory
    original_dir = Path.cwd()
    with TemporaryDirectory() as tmpdir:
        session.cd(tmpdir)
        # If you need a specific branch:
        # session.run("git", "clone", "-b", "branch-name", "https://github.com/TeamGraphix/graphix-symbolic")
        session.run("git", "clone", "https://github.com/TeamGraphix/graphix-symbolic")
        session.cd("graphix-symbolic")
        session.run("pytest")
        # Leave the directory before exiting `with` so that the
        # temporary directory can be deleted even on Windows
        session.cd(original_dir)
