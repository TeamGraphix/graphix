Installation
============

You can install Graphix from PyPI using ``pip``, or install the source code if you want to contribute to its development.

Requirements
------------

Graphix requires Python 3.10 or later. We recommend installing it in a virtual environment to keep its dependencies separate from other Python projects.

You will need:

* Python 3.10 or later
* ``pip`` for installing the package

Check your Python version with:

.. code-block:: console

   $ python --version

Installing with pip
-------------------

For most users, the recommended installation method is to install the latest published release from PyPI.

1. Create and activate a virtual environment:

   .. tab-set::

      .. tab-item:: Linux / macOS

         .. code-block:: console

            $ python -m venv .venv
            $ source .venv/bin/activate

      .. tab-item:: Windows (PowerShell)

         .. code-block:: powershell

            PS> python -m venv .venv
            PS> .venv\Scripts\Activate.ps1

2. Install Graphix:

   .. code-block:: console

      $ python -m pip install graphix

To install Graphix with its optional extra dependencies, use:

.. code-block:: console

   $ python -m pip install "graphix[extra]"

Use this option if you need the additional functionality provided by the optional dependencies.

Installing from source
----------------------

If you want to contribute to Graphix or work with the latest development version, you can install it directly from the GitHub repository.

First, clone the repository:

.. code-block:: console

   $ git clone https://github.com/TeamGraphix/graphix.git
   $ cd graphix

Graphix uses `uv <https://docs.astral.sh/uv/>`_ to manage its development environment. Install ``uv`` if necessary, then synchronize the project dependencies:

.. code-block:: console

   $ uv sync --extra dev

This creates a virtual environment and installs Graphix together with its development dependencies. To activate the environment manually, run:

.. code-block:: console

   $ source .venv/bin/activate

On Windows, use ``.venv\Scripts\Activate.ps1`` in PowerShell.

Next steps
----------

Now that Graphix is installed, you can start exploring its features:

* :doc:`getting_started` — test your installation by running your first MBQC simulation in Graphix.
* :doc:`tutorials/index` — follow practical examples and learn how to use Graphix.
* :doc:`development/apiref/index` — browse the API reference.
* `Home <landing-page>` — return to the documentation homepage.

If you encounter a problem during installation, please report it on the `Graphix issue tracker <https://github.com/TeamGraphix/graphix/issues>`_.
