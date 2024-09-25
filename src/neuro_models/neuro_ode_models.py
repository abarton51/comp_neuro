"""
This module defines base classes and models for simulating neural dynamics 
using Ordinary Differential Equations (ODEs). It includes a generic abstract 
base class `ODE_Model` for defining ODE systems, along with an implementation 
of the Quadratic Integrate-and-Fire (QIF) neuron model.

Classes:
--------
- ODE_Model(ABC): 
    Abstract base class for neural ODE models. It provides a framework for solving 
    ODEs using the Euler method and handling spiking behavior in neural systems.
    
- QIF(ODE_Model): 
    Implementation of the Quadratic Integrate-and-Fire model, which simulates 
    the dynamics of a neuron using a quadratic ODE with reset conditions for spiking.

Functions (in classes):
-----------------------
- ODE_Model.ode(y):
    Abstract method to compute the rate of change of the model's state variable (e.g., voltage).
    
- ODE_Model.euler_method(y0, t0, tn, dt):
    Solves the ODE system using the forward Euler method over the specified time period.
    
- ODE_Model.euler_method_spiking(y0, t0, tn, dt):
    Solves the ODE system using the forward Euler method with spiking behavior.

- QIF.ode(V):
    Defines the quadratic ODE for the QIF model, simulating the neuron's voltage dynamics.

- QIF.calculate_discriminant():
    Computes the discriminant for the quadratic equation governing the fixed points.
    
- QIF.calculate_critical_Ix():
    Determines the critical injected current at which the neuron behavior changes.
    
- QIF.calculate_fp():
    Computes the fixed point(s) of the system based on the injected current Ix.

- QIF.quadratic_formula():
    Solves the quadratic equation for the fixed points of the system.

- QIF.set_Ix(Ix):
    Updates the value of the injected current parameter (Ix).
"""

import numpy as np
from typing import Union
from abc import ABC
from abc import abstractmethod

class ODE_Model(ABC):
    """
    Abstract base class for simulating neural dynamics using ODEs.
    
    Provides a framework for solving ODEs with the Euler method and for handling spiking 
    in neural models. Classes that inherit from this must implement the `ode` method to 
    define the specific dynamics of the model.
    """

    def __init__(self):
        return

    # Define an abstract method for dVdt
    @abstractmethod
    def ode(self, y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Abstract method for calculating the rate of change of the 
            system's state variable (e.g., voltage).
        
        Args:
            y (float or np.ndarray): The state variable (e.g., voltage)
                of the system at the current time step.
        
        Returns:
            float or np.ndarray: The rate of change of the state variable.
        """
        pass

    # Forward Euler method implementation
    def euler_method(self, y0: float, t0: float, tn: float,
                     dt: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Solves the ODE system using the forward Euler method.

        Args:
            y0 (float): Initial value of the state variable (e.g., initial voltage).
            t0 (float): Start time.
            tn (float): End time.
            dt (float): Time step size.
        
        Returns:
            tuple: Arrays of time values and corresponding state variable values over time.
        """

        t_values = np.arange(t0, tn + dt, dt)
        y_values = np.zeros(len(t_values))
        y_values[0] = y0

        for i in range(1, len(t_values)):
            # y_{n+1} = y_n + dt * f(t_n, y_n)
            y_values[i] = y_values[i - 1] + dt * self.ode(y_values[i - 1])

        return t_values, y_values

    def euler_method_spiking(self, y0: float, t0: float, tn: float,
                             dt: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Solves the ODE system using the forward Euler method with spiking behavior.
        
        If the state variable exceeds a threshold (spiking), it is reset.

        Args:
            y0 (float): Initial value of the state variable (e.g., voltage).
            t0 (float): Start time.
            tn (float): End time.
            dt (float): Time step size.
        
        Returns:
            tuple: Arrays of time values and corresponding state variable values over time, 
                   with reset behavior for spiking neurons.
        """

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
    """
    Quadratic Integrate-and-Fire (QIF) neuron model.
    
    This model describes the dynamics of a neuron where the voltage 
    evolves quadratically and resets when it reaches a threshold. 
    It supports an injected current (Ix) as a control parameter.
    
    Attributes:
        model_name (str): Name of the model.
        param (dict): Dictionary of model parameters, including 
        injected current (Ix), membrane time constant (taum), 
        and voltage thresholds.
    """

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
        neg_b = self.V1 + self.V2
        sqrt_d = np.sqrt(self.d)
        q1 = (neg_b + sqrt_d) / 2
        q2 = (neg_b - sqrt_d) / 2
        return q1, q2

    def set_Ix(self, Ix: float):
        self.Ix = Ix
