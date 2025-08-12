# Contributing

thank you for your interest in `Graphix`!

## Motivation

The aim of `graphix` is to make the measurement-based quantum computing (MBQC) accessible by creating a one-stop environment to study and research MBQC.

## Getting started working on your contribution for `graphix`

We recommend to [fork the repository](https://docs.github.com/en/get-started/quickstart/fork-a-repo) before working on new features, whether that being an existing issue or your own idea.
Once created, you'll need to clone the repository, and you can follow below to set up the environment. You may want to set up virtual environment, such as `conda env` or `pipenv` before setting up the environment.

```bash
git clone git@github.com:<username>/graphix.git
cd graphix
pip install -e .[dev]
```

You may want to install additional packages.
Specifically, `matplotlib` is necessary to run codes in the `example` folder.

```bash
pip install matplotlib
```

For other depencencies for the docs build, see `docs/requirements.txt`.

Before comitting the code, make sure to format with `ruff`.
To format a python file, just run in the top level of the repository:

```bash
# Be sure to use the latest version of ruff
pip install -U ruff
# Sort imports and format
ruff check --select I --fix .
ruff format .
```

### Local checks

To replicate the CI pipeline locally, install `nox` and run the tests:

```bash
pip install -c requirements-dev.txt nox
nox
```

With the development dependencies installed, run the test suite explicitly:

```bash
pip install .[dev]
pytest --cov=./graphix --cov-report=xml --cov-report=term
```

### VS Code configuration

If you use VS Code for development, add a ``.vscode/settings.json`` file to
enable the linter and basic type checking on save:

```json
{
    "python.formatting.provider": "ruff",
    "editor.codeActionsOnSave": {
        "source.organizeImports": true,
        "source.fixAll": true
    },
    "python.analysis.typeCheckingMode": "basic"
}
```

and you are ready to commit the changes.

## Type-checking other Python projects that import `graphix` when installed as an editable package

If `graphix` is installed with the command given above:

```bash
pip install -e .[dev]
```

Then `mypy` fails to type-check other Python projects that import `graphix`, showing the following error:

```
error: Skipping analyzing "graphix": module is installed, but missing library stubs or py.typed marker  [import-untyped]
```

Neither `mypy` nor `pyright` detect editable packages (installed with
`pip install -e`), unless they are installed in `compat` or `strict`
mode. A workaround is to install `graphix` in `strict` mode by running
the following command in the `graphix` repository.

```bash
pip install -e . --config-settings editable_mode=strict
```

See https://github.com/pypa/setuptools/issues/3518 for more details.
