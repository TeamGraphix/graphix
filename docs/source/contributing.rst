Contributing to Graphix
=======================

This page summarises useful tips for developing the project locally.

Setup
-----

Clone the repository and install all development dependencies::

    git clone https://github.com/TeamGraphix/graphix.git
    cd graphix
    uv sync --extra dev --extra extra

Local checks
------------

Run the full CI suite locally::

    uv run nox

Or run individual checks::

    uv run ruff check .
    uv run ruff format --check .
    uv run mypy
    uv run pytest

Format code before committing::

    uv run ruff check --select I --fix .
    uv run ruff format .

VS Code configuration
---------------------

Using `VS Code <https://code.visualstudio.com/>`_ helps catch issues early.  A
minimal ``.vscode/settings.json`` may look like::

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

These settings enable the linter, format the code on save and turn on basic
static type checking through the Pylance extension.