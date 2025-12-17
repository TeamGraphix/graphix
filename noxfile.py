"""Run tests with nox."""

from __future__ import annotations

from tempfile import TemporaryDirectory

import nox
from nox import Session


def install_pytest(session: Session) -> None:
    """Install pytest when requirements-dev.txt is not installed."""
    session.install("pytest", "pytest-mock", "pytest-benchmark", "pytest-mpl", "psutil")


def run_pytest(session: Session, doctest_modules: bool = False, mpl: bool = False) -> None:
    """Run pytest."""
    args = ["pytest"]
    if doctest_modules:
        args.append("--doctest-modules")
    if mpl:
        args.append("--mpl")
    session.run(*args)


@nox.session(python=["3.10", "3.11", "3.12", "3.13"])
def tests_minimal(session: Session) -> None:
    """Run the test suite with minimal dependencies."""
    session.install(".")
    install_pytest(session)
    run_pytest(session, mpl=True)


@nox.session(python=["3.10", "3.11", "3.12", "3.13"])
def tests_dev(session: Session) -> None:
    """Run the test suite with dev dependencies."""
    session.install(".[dev]")
    # We cannot run `pytest --doctest-modules` here, since some tests
    # involve optional dependencies, like pyzx.
    run_pytest(session, mpl=True)


@nox.session(python=["3.10", "3.11", "3.12", "3.13"])
def tests_extra(session: Session) -> None:
    """Run the test suite with extra dependencies."""
    session.install(".[extra]")
    install_pytest(session)
    session.install("nox")  # needed for `--doctest-modules`
    run_pytest(session, doctest_modules=True)


@nox.session(python=["3.10", "3.11", "3.12", "3.13"])
def tests_all(session: Session) -> None:
    """Run the test suite with all dependencies."""
    session.install(".[dev,extra]")
    run_pytest(session, doctest_modules=True, mpl=True)


@nox.session(python=["3.10", "3.11", "3.12", "3.13"])
def tests_symbolic(session: Session) -> None:
    """Run the test suite of graphix-symbolic."""
    session.install(".")
    install_pytest(session)
    session.install("nox")  # needed for `--doctest-modules`
    # Use `session.cd` as a context manager to ensure that the
    # working directory is restored afterward. This is important
    # because Windows cannot delete a temporary directory while it
    # is the working directory.
    with TemporaryDirectory() as tmpdir, session.cd(tmpdir):
        # If you need a specific branch:
        # session.run("git", "clone", "-b", "branch-name", "https://github.com/TeamGraphix/graphix-symbolic")
        session.run("git", "clone", "https://github.com/TeamGraphix/graphix-symbolic")
        with session.cd("graphix-symbolic"):
            run_pytest(session, doctest_modules=True)


@nox.session(python=["3.10", "3.11", "3.12", "3.13"])
def tests_qasm_parser(session: Session) -> None:
    """Run the test suite of graphix-qasm-parser."""
    session.install(".")
    install_pytest(session)
    session.install("nox")  # needed for `--doctest-modules`
    # Use `session.cd` as a context manager to ensure that the
    # working directory is restored afterward. This is important
    # because Windows cannot delete a temporary directory while it
    # is the working directory.
    with TemporaryDirectory() as tmpdir, session.cd(tmpdir):
        # See https://github.com/TeamGraphix/graphix-qasm-parser/pull/7
        session.run("git", "clone", "-b", "graphix_master", "https://github.com/TeamGraphix/graphix-qasm-parser")
        with session.cd("graphix-qasm-parser"):
            session.install(".")
            run_pytest(session, doctest_modules=True)
