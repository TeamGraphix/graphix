Visualization tool
==================

:mod:`graphix.visualization` module
+++++++++++++++++++++++++++++++++++++++++

This module provides functions to visualize the resource state of MBQC pattern.
If flow or gflow exist, the tool take them into account and show the information flow as directed edges.

.. currentmodule:: graphix.visualization

.. autoclass:: GraphVisualizer
    :members:

:mod:`graphix.pretty_print` module
+++++++++++++++++++++++++++++++++++

This modules provides functions to format patterns and flows.

.. currentmodule:: graphix.pretty_print

.. autoclass:: OutputFormat

.. autofunction:: angle_to_str

.. autofunction:: command_to_str

.. autofunction:: pattern_to_str

.. autofunction:: flow_to_str

.. autofunction:: xzcorr_to_str
