# Contributing

thank you for your interest in `Graphix`!

## Motivation

The aim of `graphix` is to make the measurement-based quantum computing (MBQC) accessible by creating a one-stop environment to study and research MBQC.

## Getting started working on your contribution for `graphix`

We recommend to [fork the repository](https://docs.github.com/en/get-started/quickstart/fork-a-repo) before working on new features, whether that being an existing issue or your own idea.
Once created, you'll need to clone the repository, and you can follow below to set up the environment. We recommend you use a virtual environment like `uv`, but you may prefer to use `conda env` or `pipenv`.

```bash
git clone git@github.com:<username>/graphix.git
cd graphix
uv sync --extra dev --extra extra
```

This creates a virtual environment and installs all development dependencies (linting, testing, type-checking, etc.) from the lockfile.

## Local checks

To run individual checks:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

You can also run the full CI suite locally:
```bash
uv run nox
```

Before committing, format the code:

```bash
uv run ruff check --select I --fix .
uv run ruff format .
```

### VS Code configuration

If you use VS Code for development, add a ``.vscode/settings.json`` file to
enable the linter and basic type checking on save:

```json
{
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.codeActionsOnSave": {
      "source.organizeImports.ruff": "explicit",
      "source.fixAll.ruff": "explicit"
    }
  },
  "python.analysis.typeCheckingMode": "basic"
}
```

## Type-checking other Python projects that import `graphix` when installed as an editable package

If `graphix` is installed with `uv sync` or `pip install -e`, `mypy` may fail to type-check other Python projects that import `graphix`, showing the following error:

```
error: Skipping analyzing "graphix": module is installed, but missing library stubs or py.typed marker  [import-untyped]
```

To work around this, install `graphix` non-editable:

```bash
uv sync --no-editable
```

Or with pip:

```bash
pip install . --config-settings editable_mode=strict
```

See https://github.com/pypa/setuptools/issues/3518 for more details.
