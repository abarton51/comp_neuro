import numpy as np
from typing import Union
from abc import ABC
from abc import abstractmethod

class ODE_Model(ABC):
    def __init__(self):
        return
    
    # Define an abstract method for dVdt
    @abstractmethod
    def ode(self, y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Method to calculate the rate of change of voltage.
        Must be implemented in any subclass."""
        pass
    
    # Forward Euler method implementation
    def euler_method(self, y0: float, t0: float, tn: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
        t_values = np.arange(t0, tn + dt, dt)
        y_values = np.zeros(len(t_values))
        y_values[0] = y0

        for i in range(1, len(t_values)):
            # y_{n+1} = y_n + dt * f(t_n, y_n)
            y_values[i] = y_values[i - 1] + dt * self.ode(y_values[i - 1])

        return t_values, y_values

    def euler_method_spiking(self, y0: float, t0: float, tn: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
        t_values = np.arange(t0, tn + dt, dt)
        y_values = np.zeros(len(t_values))
        y_values[0] = y0

        for i in range(1, len(t_values)):
            # y_{n+1} = y_n + dt * f(t_n, y_n)
            y_values[i] = y_values[i - 1] + dt * self.ode(y_values[i - 1])
            if y_values[i] >= self.yth:  # Spike condition
                y_values[i] = self.yr  # Reset after spike

        return t_values, y_values
    
        
class QIF(ODE_Model):
    model_name = "Quadratic Integrate-and-Fire Model"
    
    def __init__(self, param: dict):
        super().__init__()
        
        # Ensure 'param' is a dictionary
        if not isinstance(param, dict):
            raise TypeError("Expected a dictionary for 'param'")
        
        # Check for any None values in the dictionary
        for key, value in param.items():
            if key != 'Ix':
                if value is None:
                    raise ValueError(f"Parameter value for '{key}' cannot be None")
        
        # Injected current/ Control parameter
        self.Ix=param.get('Ix', None)
        
        # Fixed parameters
        self.taum=param['taum']
        self.c=param['c']
        self.V1=param['V1']
        self.V2=param['V2']
        # Threshold voltage for spiking
        self.yth=param['Vth']
        # Reset voltage for spiking
        self.yr=param['Vr']
        
        self.d = self.calculate_discriminant() if self.Ix else None
        self.critical_Ix = self.calculate_critical_Ix()
        self.fp = self.calculate_fp() if self.Ix else None
        
    def ode(self, V: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return (1 / self.taum) * (self.c * (V - self.V1) * (V - self.V2) + self.Ix)
    
    def calculate_discriminant(self):
        return (self.V1 + self.V2)**2 - 4 * (self.V1*self.V2 + self.Ix / self.c)
    
    def calculate_critical_Ix(self):
        return self.c * ((self.V1 + self.V2)**2 / 4 - self.V1 * self.V2)
    
    def calculate_fp(self) -> Union[float, tuple[float, float]]:
        if self.Ix < self.critical_Ix:
            return None
        elif self.Ix == self.critical_Ix:
            return (self.V1 + self.V2) / 2
        else:
            return self.quadratic_formula()
    
    def quadratic_formula(self) -> tuple[float, float]:
        neg_b = (self.V1 + self.V2)
        sqrt_d = np.sqrt(self.d)
        q1 = (neg_b + sqrt_d) / 2
        q2 = (neg_b - sqrt_d) / 2
        return q1, q2
    
    def set_Ix(self, Ix: float):
        self.Ix = Ix