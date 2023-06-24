"""Patrameter class for graphix
Parameter object acts as a placeholder of measurement angles and 
allows the manipulation of the measurement pattern without specific value assignment.

"""
import numpy as np


class Parameter:
    """Placeholder for measurement angles, which allows the pattern optimizations
    without specifying measurement angles for measurement commands.
    
    Either use for rotation gates of :class:`Circuit` class or for 
    the measurement angle of the measurement commands to be added with :meth:`Pattern.add` method.
    
    Example:
    .. code-block:: python

        # rotation gate
        from graphix import Circuit
        circuit = Circuit(1)
        circuit.rx(0, Parameter('r1'))
        pattern = circuit.transpile()
        val = {'r1': 0.5}
        # simulate with parameter assignment
        sv = pattern.simulate_pattern(backend='statevector', params=val)
        sv2 = circuit.simulate_statevector(params=val)

    """

    def __init__(self, name):
        """Create a new :class:`Parameter` object.

        Parameters
        ----------
        name : str
            name of the parameter, used for binding values.
        """
        assert isinstance(name, str)
        self._name = name
        self._value = np.nan
        self._assigned = False

    def __repr__(self):
        return f"graphix.Parameter, name={self.name}"
    
    @property
    def name(self):
        return self._name

    @property
    def assigned(self):
        return self._assigned

    def bind(self, value, unit="pi"):
        """Binds the parameters to itself.
        
        Parameters
        ----------
        values : float
            value to assign to the parameter.
        unit : "pi" or "radian"
            unit of the rotation angle, either in rad/pi or rad.
        """
        assert not self._assigned
        assert isinstance(value, float) or isinstance(value, int)
        if unit == "pi":
            self.value = value # rotation angles are in unit of pi for measurement commands
        elif unit == "radian":
            self.value = value / np.pi
        else:
            raise ValueError(f"Unknown unit {unit}")
        self._assigned = True

    def bind_from_dict(self, values, unit="pi"):
        """Binds the parameters to itself.
        
        Parameters
        ----------
        values : float
            value to assign to the parameter.
        unit : "pi" or "radian"
            unit of the rotation angle, either in rad/pi or rad.
        """
        assert isinstance(values, float)
        if self.name in values.keys():
            self.bind(values[self._name])
        else:
            raise ValueError(f"{self._name} not found")

    def __mul__(self, other):
        """mul magic function is used to return measurement angles in simulators."""
        assert self._assigned, "parameter cannot be used for calculation unless value is assigned."
        assert isinstance(other, float)
        return self.value * other

    def __mod__(self, other):
        """mod magic function returns nan so that evaluation of 
        mod of measurement angles in :meth:`graphix.pattern.is_pauli_measurement`
        will not cause error. returns nan so that this will not be considered Pauli measurement.
        """
        assert isinstance(other, float)
        return np.nan