
Quantum channels and noise models
+++++++++++++++++++++++++++++++++

Kraus channel
-------------

.. currentmodule:: graphix.channels

.. autoclass:: KrausChannel
    :members:

.. autofunction:: dephasing_channel

.. autofunction:: depolarising_channel

.. autofunction:: pauli_channel

.. autofunction:: two_qubit_depolarising_channel

.. autofunction:: two_qubit_depolarising_tensor_channel


Noise model classes
-------------------


.. currentmodule:: graphix.noise_models.noise_model

.. autoclass:: Noise
    :members:

.. autoclass:: A
    :members:

.. autoclass:: NoiseModel
    :members:

.. autoclass:: ComposeNoiseModel
    :members:

.. currentmodule:: graphix.noise_models.noiseless_noise_model

.. autoclass:: NoiselessNoiseModel
    :members:

.. currentmodule:: graphix.noise_models.depolarising_noise_model

.. autoclass:: DepolarisingNoise
    :members:

.. autoclass:: TwoQubitDepolarisingNoise
    :members:

.. autoclass:: DepolarisingNoiseModel
    :members:
