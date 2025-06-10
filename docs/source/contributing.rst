Contributing to Graphix
=======================

This page summarises useful tips for developing the project locally.

Local checks
------------

* Run ``nox -s tests`` to execute the test suite.  This mirrors what the
  continuous integration service runs.
* Format the code with :command:`ruff` before committing::

    ruff check --select I --fix .
    ruff format .

Additional commands from the CI configuration are useful for replicating
the testing environment locally::

    pip install -c requirements-dev.txt nox
    nox --python 3.12

    pip install .[dev]
    pytest --cov=./graphix --cov-report=xml --cov-report=term

VS Code configuration
---------------------

Using `VS Code <https://code.visualstudio.com/>`_ helps catch issues early.  A
minimal ``.vscode/settings.json`` may look like::

    {
        "python.formatting.provider": "ruff",
        "editor.codeActionsOnSave": {
            "source.organizeImports": true,
            "source.fixAll": true
        },
        "python.analysis.typeCheckingMode": "basic"
    }

These settings enable the linter, format the code on save and turn on basic
static type checking through the Pylance extension.
