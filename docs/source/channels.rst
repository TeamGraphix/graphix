
Quantum channels and noise models
+++++++++++++++++++++++++++++++++

Kraus channel
-------------

.. currentmodule:: graphix.channels

.. autoclass:: KrausChannel
    :members:

.. autofunction:: dephasing_channel

.. autofunction:: depolarising_channel

.. autofunction:: amplitude_damping_channel

.. autofunction:: pauli_channel

.. autofunction:: two_qubit_depolarising_channel

.. autofunction:: two_qubit_amplitude_damping_channel

.. autofunction:: two_qubit_depolarising_tensor_channel


Noise model classes
-------------------


.. currentmodule:: graphix.noise_models.noise_model

.. autoclass:: Noise
    :members:

.. autoclass:: ApplyNoise
    :members:

.. autoclass:: NoiseModel
    :members:

.. autoclass:: NoiselessNoiseModel
    :members:

.. autoclass:: ComposeNoiseModel
    :members:

.. currentmodule:: graphix.noise_models.depolarising

.. autoclass:: DepolarisingNoise
    :members:

.. autoclass:: TwoQubitDepolarisingNoise
    :members:

.. autoclass:: DepolarisingNoiseModel
    :members:

.. currentmodule:: graphix.noise_models.amplitude_damping

.. autoclass:: AmplitudeDampingNoise
    :members:

.. autoclass:: TwoQubitAmplitudeDampingNoise
    :members:

.. autoclass:: AmplitudeDampingNoiseModel
    :members:
