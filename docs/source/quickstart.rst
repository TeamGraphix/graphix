Quickstart
===========

This section explains how to prepare a Python 3 environment for using graphix.

Package Installation
--------------------

Please make sure you have pip version 19.3 or higher, and update it if necessary.

.. code-block:: bash

    $ pip --version
    pip 18.1.1
    $ pip install --upgrade pip
    ...
    Successfully installed pip-20.1.1

In some environments, the Python3 `pip` command is provided as the `pip3` command. If this is the case, please replace it accordingly.

Graphix can be installed with the following command:

.. code-block:: bash

    $ pip install graphix


Update Package
--------------

You can perform a package update in the following way:

.. code-block:: bash

    $ pip install --upgrade graphix

Import Package
--------------

If the installation is successful, graphix can be imported in your Python script as shown below:

.. code-block:: python

    import graphix

>>> graphix.__version__    # Check the version
'0.0.1'

You can follow the :doc:`tutorial` to learn how to design and simulate MBQC using graphix library.
