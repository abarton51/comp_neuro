"""
Author: Austin Barton
Collaborators: None
Credits: **ALL** code was written, developed, and used solely by Austin Barton.
    A large majority of the documentation was generated using the Chat-GPT4
    LLM after code was developed to make it easier to come back to and read.

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
                             dt: float, yth: float, yr: float) -> tuple[np.ndarray, np.ndarray]:
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
            if y_values[i] >= yth:  # Spike condition
                y_values[i] = yr  # Reset after spike

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

        # Parameters
        self.Ix=param.get('Ix', None)
        self.taum=param['taum']
        self.c=param['c']
        self.V1=param['V1']
        self.V2=param['V2']
        # Threshold voltage for spiking
        self.Vth=param.get('Vth', None)
        # Reset voltage for spiking
        self.Vr=param.get('Vr', None)

        self.discriminant = self.calculate_discriminant() if self.Ix else None
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
        sqrt_d = np.sqrt(self.discriminant)
        q1 = (neg_b + sqrt_d) / 2
        q2 = (neg_b - sqrt_d) / 2
        return q1, q2

    def set_Ix(self, Ix: float):
        self.Ix = Ix

class Iz_Simple(ODE_Model):

    model_name = "Izhikevich Simple Model"

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

        # Parameters
        self.a=param['a']
        self.b=param['b']
        self.c=param['c']
        self.d=param['d']
        # Threshold voltage for spiking
        self.Vth=param.get('Vth', None)
        self.nth=param.get('nth', None)
        # Reset voltage for spiking
        self.Vr=param.get('Vr', None)
        self.nr=param.get('nr', None)

        self.discriminant = self.calculate_discriminant() if self.Ix else None
        self.critical_Ix = self.calculate_critical_Ix() if self.Ix else None
        self.fp = self.calculate_fp() if self.Ix else None

    def ode(self,
            input: np.ndarray[Union[float, np.ndarray], Union[float, np.ndarray]]) \
                -> np.ndarray[Union[float, np.ndarray], Union[float, np.ndarray]]:
        return np.asarray([self.dvdt(input), self.dndt(input)])

    def dvdt(self,
            input: np.ndarray[Union[float, np.ndarray], Union[float, np.ndarray]]) \
                -> np.ndarray[Union[float, np.ndarray], Union[float, np.ndarray]]:
                    
        return 0.04 * input[0]**2 + 5 * input[0] - input[1] + 140 + self.Ix

    def dndt(self,
            input: np.ndarray[Union[float, np.ndarray], Union[float, np.ndarray]]) \
                -> np.ndarray[Union[float, np.ndarray], Union[float, np.ndarray]]:
        return self.a * (self.b * input[0] - input[1])

    def euler_method_spiking(self, V0: float, n0: float, t0: float, tn: float, dt: float,
                             Vth: float) \
                                -> np.ndarray[np.ndarray, np.ndarray]:
        t_values = np.arange(t0, tn + dt, dt)
        y_values = np.zeros((2, len(t_values)))
        y_values[0,0] = V0
        y_values[1,0] = n0

        for i in range(1, len(t_values)):
            # y_{n+1} = y_n + dt * f(t_n, y_n)
            y_values[:,i] = y_values[:,i - 1] + dt * self.ode(y_values[:,i - 1])
            if y_values[0,i] >= Vth:  # Spike condition
                y_values[0,i] = self.c  # Reset after spike
                y_values[1,i] = y_values[1,i-1] + self.d

        return t_values, y_values

    def set_Vth(self, Vth: float):
        self.Vth = Vth

    def set_nth(self, nth: float):
        self.nth = nth

    def set_Ix(self, Ix: float):
        self.Ix = Ix
