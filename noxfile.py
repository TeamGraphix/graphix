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

nox.options.default_venv_backend = "uv|virtualenv"

PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]


def install_pytest(session: Session) -> None:
    """Install pytest when the dev extra is not installed."""
    session.install("pytest", "pytest-mock", "pytest-benchmark", "pytest-mpl", "psutil")


def run_pytest(session: Session, *args: str, doctest_modules: bool = False, mpl: bool = False) -> None:
    """Run pytest."""
    cmdline = ["pytest", "-W", "error", *args]
    if doctest_modules:
        cmdline.append("--doctest-modules")
    if mpl:
        cmdline.append("--mpl")
    session.run(*cmdline)


@nox.session(python=PYTHON_VERSIONS)
def tests_minimal(session: Session) -> None:
    """Run the test suite with minimal dependencies."""
    session.install(".")
    install_pytest(session)
    # We cannot run `pytest --mpl` here because the visualization output
    # depends on the Matplotlib version, and the pinning is only present
    # in the `dev` extra.
    run_pytest(session)


@nox.session(python=PYTHON_VERSIONS)
def tests_all(session: Session) -> None:
    """Run the test suite with all dependencies."""
    session.install(".[dev]")
    # This dependency is added here to avoid circular dependencies
    session.install("-r", ".github/qasm-parser-requirements.txt")
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
    initialization: Callable[[Session], bool | None] | None = None
    install_target: str = "."
    pytest_args: tuple[str, ...] = ()


REVERSE_DEPENDENCIES = {
    "graphix-symbolic": ReverseDependency("https://github.com/TeamGraphix/graphix-symbolic"),
    "graphix-stim-backend": ReverseDependency("https://github.com/TeamGraphix/graphix-stim-backend"),
    "graphix-qasm-parser": ReverseDependency("https://github.com/TeamGraphix/graphix-qasm-parser"),
    "graphix-ibmq": ReverseDependency("https://github.com/TeamGraphix/graphix-ibmq", doctest_modules=False),
    "graphix-stim-compiler": ReverseDependency("https://github.com/TeamGraphix/graphix-stim-compiler"),
    "graphix-pyzx": ReverseDependency(
        "https://github.com/TeamGraphix/graphix-pyzx",
        # Precompile pyzx before running pytest with warnings-as-errors.
        # See zxcalc/pyzx#518.
        initialization=lambda session: session.run("python", "-c", "import pyzx"),
        # Filter warnings raised by pyzx (zxcalc/pyzx#509)
        pytest_args=(
            "-W",
            "ignore:In 3.13 classes created inside an enum will not become a member.:DeprecationWarning",
        ),
    ),
    "veriphix": ReverseDependency(
        "https://github.com/qat-inria/veriphix", doctest_modules=False, install_target=".[dev]"
    ),
    "graphix-mqtbench": ReverseDependency("https://github.com/TeamGraphix/graphix-mqtbench", branch="refs/pull/5/head"),
}


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("package_name", list(REVERSE_DEPENDENCIES))
def tests_reverse_dependencies(session: Session, package_name: str) -> None:
    """Run the test suite of reverse dependencies."""
    package = REVERSE_DEPENDENCIES[package_name]
    url = urlparse(package.repository)
    dirname = Path(url.path).name
    assert isinstance(session.python, str)
    if package.version_constraint is not None and Version(session.python) not in package.version_constraint:
        session.skip(
            f"{dirname} only supports Python versions {package.version_constraint}; current Python version: {session.python}"
        )

    install_pytest(session)
    if package.doctest_modules:
        session.install("nox")
    with TemporaryDirectory() as tmpdir:
        with session.cd(tmpdir):
            session.run("git", "clone", package.repository, external=True)
            with session.cd(dirname):
                if package.branch is not None:
                    # Use `git fetch` instead of `git clone -b` to support
                    # special refs such as `refs/pull/N/head`
                    session.run("git", "fetch", "origin", package.branch, external=True)
                    session.run("git", "checkout", "--detach", "FETCH_HEAD", external=True)
                # graphix installation fails without constraint on numba
                session.install(package.install_target, "numba>=0.65.1")
        # Note that `session.cd` is used as a context manager above,
        # so that the working directory is restored at this point.  We
        # install now the graphix package from the working directory.
        # This is done after having installed the reverse dependency,
        # so that we run the test with the current graphix codebase,
        # even if another graphix version has been pinned in the
        # reverse dependendy.
        session.install(".[dev]")
        # Use `session.cd` as a context manager again to ensure that the
        # working directory is restored afterward. This is important
        # because Windows cannot delete a temporary directory while it
        # is the working directory.
        with session.cd(tmpdir), session.cd(dirname):
            if package.initialization is not None:
                package.initialization(session)
            run_pytest(session, *package.pytest_args, doctest_modules=package.doctest_modules)
