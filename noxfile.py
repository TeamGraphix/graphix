"""Run tests with nox."""

from __future__ import annotations

from tempfile import TemporaryDirectory

import nox
from nox import Session


def install_pytest(session: Session) -> None:
    """Install pytest when requirements-dev.txt is not installed."""
    session.install("pytest", "pytest-mock", "pytest-benchmark", "psutil")


@nox.session(python=["3.9", "3.10", "3.11", "3.12", "3.13"])
def tests_minimal(session: Session) -> None:
    """Run the test suite with minimal dependencies."""
    session.install("-e", ".")
    install_pytest(session)
    # We cannot run `pytest --doctest-modules` here, since some tests
    # involve optional dependencies, like pyzx.
    session.run("pytest")


# Note that recent types-networkx versions don't support Python 3.9
@nox.session(python=["3.10", "3.11", "3.12", "3.13"])
def tests_dev(session: Session) -> None:
    """Run the test suite with dev dependencies."""
    session.install("-e", ".[dev]")
    # We cannot run `pytest --doctest-modules` here, since some tests
    # involve optional dependencies, like pyzx.
    session.run("pytest")


@nox.session(python=["3.9", "3.10", "3.11", "3.12", "3.13"])
def tests_extra(session: Session) -> None:
    """Run the test suite with extra dependencies."""
    session.install("-e", ".[extra]")
    install_pytest(session)
    session.install("nox")  # needed for `--doctest-modules`
    session.run("pytest", "--doctest-modules")


@nox.session(python=["3.10", "3.11", "3.12", "3.13"])
def tests_all(session: Session) -> None:
    """Run the test suite with all dependencies."""
    session.install("-e", ".[dev,extra]")
    session.run("pytest", "--doctest-modules")


@nox.session(python=["3.9", "3.10", "3.11", "3.12", "3.13"])
def tests_symbolic(session: Session) -> None:
    """Run the test suite of graphix-symbolic."""
    session.install("-e", ".")
    install_pytest(session)
    session.install("nox")  # needed for `--doctest-modules`
    # Use `session.cd` as a context manager to ensure that the
    # working directory is restored afterward. This is important
    # because Windows cannot delete a temporary directory while it
    # is the working directory.
    with TemporaryDirectory() as tmpdir, session.cd(tmpdir):
        # If you need a specific branch:
        # session.run("git", "clone", "-b", "branch-name", "https://github.com/TeamGraphix/graphix-symbolic")
        # See https://github.com/TeamGraphix/graphix-symbolic/pull/4
        session.run("git", "clone", "-b", "branch_selector", "https://github.com/thierry-martinez/graphix-symbolic")
        # session.run("git", "clone", "https://github.com/TeamGraphix/graphix-symbolic")
        with session.cd("graphix-symbolic"):
            session.run("pytest", "--doctest-modules")
