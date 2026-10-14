Contributing to Graphix
=======================

Graphix is an open-source software library, which means we always welcome contributions from the user community. If you are interested in helping improve Graphix, here are some useful tips for developing the project locally.

Setup your local environment
----------------------------

Clone the repository and install all development dependencies:

.. code-block:: console

    $ git clone https://github.com/TeamGraphix/graphix.git
    $ cd graphix
    $ pip install uv
    $ uv sync --extra dev --extra extra

Run local checks
------------

Run the full CI suite on your machine:

.. code-block:: console

    $ uv run nox

Or run individual checks:

.. code-block:: console

    $ uv run ruff check .
    $ uv run ruff format --check .
    $ uv run mypy
    $ uv run pytest

Format code and check code coverage before committing:

.. code-block:: console

    $ uv run ruff check --select I --fix .
    $ uv run ruff format .
    $ uv run pytest --cov=./graphix --cov-report=xml --cov-report=term --doctest-modules

VS Code configuration
---------------------

Using `VS Code <https://code.visualstudio.com/>`_ helps catch issues early.  A minimal ``.vscode/settings.json`` may look like:

.. code-block:: json

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

These settings enable the linter, format the code on save and turn on basic static type-checking through the Pylance extension.

Other ways to get involved
--------------------------

If you have a specific idea to improve Graphix, we encourage you to open an issue on the Graphix `Issue tracker <https://github.com/TeamGraphix/graphix/issues>`_ and the development team will take a look at implementing it. 

Alternatively, for more general ideas about new features, help using Graphix or questions about MBQC theory, you can start a conversation on the `Discussions page <https://github.com/TeamGraphix/graphix/discussions>`_ on GitHub or get in touch with us on the Graphix `Slack channel <https://graphix-org.slack.com>`_.