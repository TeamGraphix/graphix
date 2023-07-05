"""Patrameter class for graphix
Parameter object acts as a placeholder of measurement angles and 
allows the manipulation of the measurement pattern without specific value assignment.

"""
import numpy as np
import sympy as sp

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
        self._value = None
        self._assignment = None
        self._expression = sp.Symbol(name= name)

    def __repr__(self):
        if self.assigned:
            return f"expr = {self._expression}, assignment = [ {self.parameters.pop()} : {self._assignment} ]"
        else:
            return f"expr = {self._expression} "
        
    
    @property
    def name(self):
        return self._name

    @property
    def assigned(self):
        return isinstance(self._assignment, (int, float))
    
    @property
    def expression(self):
        return self._expression
    
    @property
    def value(self):
        return self._value
    
    @property
    def parameters(self):
        return self._expression.free_symbols
    

    def bind_value(self, value, overwrite= False):
        """Binds the parameters to itself.
        
        Parameters
        ----------
        values_dict : float
            dict of values to assign to the parameter.

        """
        
        # assert isinstance( value_dict , dict)
        if len(self._expression.free_symbols) == 0:
            raise ValueError(" No unassinged symbols in self.expression")
        
        symbol = self._expression.free_symbols.pop()
        val = self._expression.subs({symbol: value})

        if len(val.free_symbols) == 0:

            if overwrite:
                self._value = float(val)
                self._assignment = value
            else :
                if not self.assigned:
                    self._value = float(val)
                    self._assignment = value
                else :
                    raise ValueError("Symbols are already assigned, set overwrite = True to overwrite value")
        
        if len(val.free_symbols) != 0:
            print(" WARNING: all symbols in self.expression is not assigned, remaining variables : " 
                  + str(val.free_symbols))
            self._expression = val
        
        return self
    
    def bind(self, parameter_map, allow_unknown_parameters= True, overwrite= False):

        for parameter, value in parameter_map.items():
            
            if parameter.parameters.issubset(self.parameters):
                # print("binding-parametrs") ##check
                self.bind_value(value, overwrite= overwrite)
            
        return self

    def __mul__(self, other):
        self._expression = self._expression * other
        return self
    
    def __rmul__(self, other):
        self._expression = other * self._expression
        return self
    
    def __add__(self, other):
        # return self._expression + other
        self._expression = self._expression + other
        return self
    
    def __radd__(self, other):
        self._expression = other + self._expression
        return self
    
    def __sub__(self, other):
        self._expression = self._expression - other
        return self
    
    def __rsub__(self, other):
        self._expression = other - self._expression
        return self
    
    def __neg__(self):
        self._expression = self._expression * -1.0 
        return self
    
    def __truediv__(self, other):
        self._expression = self._expression / other
        return self
    
    def __rtruediv__(self, other):
        self._expression = other / self._expression
        return self

    def __mod__(self, other):
        """mod magic function returns nan so that evaluation of 
        mod of measurement angles in :meth:`graphix.pattern.is_pauli_measurement`
        will not cause error. returns nan so that this will not be considered Pauli measurement.
        """
        assert isinstance(other, float) or isinstance(other, int)
        return np.nan
    
    def sin(self):
        self._expression = sp.sin(self._expression)
        return self
    
    def cos(self):
        self._expression = sp.cos(self._expression)
        return self
    
    def tan(self): 
        self._expression = sp.tan(self._expression)
        return self
    
    def arcsin(self):
        self._expression = sp.asin(self._expression) 
        return self
    
    def arccos(self):
        self._expression = sp.acos(self._expression)
        return self
    
    def arctan(self):
        self._expression = sp.atan(self._expression)
        return self
    
    def exp(self):
        self._expression = sp.exp(self._expression)
        return self
    
    def log(self):
        self._expression = sp.log(self._expression)
        return self


     