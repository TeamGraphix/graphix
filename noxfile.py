"""Run tests with nox."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import nox
from nox import Session
from packaging.version import Version

if TYPE_CHECKING:
    from collections.abc import Callable

nox.options.default_venv_backend = "venv"
PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]


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


@nox.session(python=PYTHON_VERSIONS)
def tests_minimal(session: Session) -> None:
    """Run the test suite with minimal dependencies."""
    session.install(".")
    install_pytest(session)
    run_pytest(session, mpl=True)


@nox.session(python=PYTHON_VERSIONS)
def tests_dev(session: Session) -> None:
    """Run the test suite with dev dependencies."""
    session.install(".[dev]")
    # We cannot run `pytest --doctest-modules` here, since some tests
    # involve optional dependencies, like pyzx.
    run_pytest(session, mpl=True)


@nox.session(python=PYTHON_VERSIONS)
def tests_extra(session: Session) -> None:
    """Run the test suite with extra dependencies."""
    session.install(".[extra]")
    install_pytest(session)
    session.install("nox")  # needed for `--doctest-modules`
    run_pytest(session, doctest_modules=True)


@nox.session(python=PYTHON_VERSIONS)
def tests_all(session: Session) -> None:
    """Run the test suite with all dependencies."""
    session.install(".[dev,extra]")
    run_pytest(session, doctest_modules=True, mpl=True)


@dataclass
class VersionRange:
    """Version range."""

    lower: Version | None = field(default=None, metadata={"description": "Lower bound (inclusive)"})
    upper: Version | None = field(default=None, metadata={"description": "Upper bound (exclusive)"})

    def __contains__(self, item: object) -> object:
        """Test whether item is in range."""
        if not isinstance(item, Version):
            return NotImplemented
        return (self.lower is None or self.lower <= item) and (self.upper is None or item < self.upper)

    def __str__(self) -> str:
        """Return a string representation of the range."""
        if self.lower is not None:
            if self.upper is not None:
                return f"between {self.lower} (inclusive) and {self.upper} (exclusive)"
            return f"from {self.lower} (inclusive)"
        if self.upper is not None:
            return f"up to {self.upper} (exclusive)"
        return "no range"


@dataclass
class ReverseDependency:
    """Reverse dependency definition."""

    repository: str
    branch: str | None = None
    version_constraint: VersionRange | None = None
    doctest_modules: bool = True
    initialization: Callable[[Session], None] | None = None


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize(
    "package",
    [
        ReverseDependency(
            "https://github.com/thierry-martinez/graphix-stim-backend",
            branch="fix_graphix423",
            version_constraint=VersionRange(upper=Version("3.14")),
        ),
        ReverseDependency(
            # "https://github.com/TeamGraphix/graphix-symbolic",
            "https://github.com/thierry-martinez/graphix-symbolic",
            branch="fix/graphix_220",
            version_constraint=VersionRange(upper=Version("3.14")),
        ),
        ReverseDependency("https://github.com/TeamGraphix/graphix-qasm-parser", branch="fix_angles"),
        ReverseDependency(
            "https://github.com/thierry-martinez/veriphix",
            branch="graphix_181",
            version_constraint=VersionRange(upper=Version("3.14")),
            doctest_modules=False,
            initialization=lambda session: session.run("python", "-m", "veriphix.sampling_circuits.experiments"),
        ),
        ReverseDependency(
            "https://github.com/thierry-martinez/graphix-ibmq",
            branch="fix/graphix_423",
            version_constraint=VersionRange(upper=Version("3.14")),
            doctest_modules=False,
        ),
    ],
)
def tests_reverse_dependencies(session: Session, package: ReverseDependency) -> None:
    """Run the test suite of reverse dependencies."""
    url = urlparse(package.repository)
    dirname = Path(url.path).name
    # Check version constraints
    assert isinstance(session.python, str)
    if package.version_constraint and Version(session.python) not in package.version_constraint:
        session.skip(f"Skipping: {dirname} version mismatch.")
    # Install dev tools needed for the test run
    session.install(".[dev]")
    with TemporaryDirectory() as tmpdir:
        with session.cd(tmpdir):
            branch_args = ["-b", package.branch] if package.branch else []
            session.run("git", "clone", *branch_args, package.repository)
            with session.cd(dirname):
                session.install(".")
        # Re-install local graphix to ensure we test the current PR/branch code
        session.install(".")
        with session.cd(tmpdir), session.cd(dirname):
            if package.initialization:
                package.initialization(session)
            run_pytest(session, doctest_modules=package.doctest_modules)
