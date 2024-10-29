"""
Author: Austin Barton
Collaborators: None
Credits: **ALL** code was written, developed, and used solely by Austin Barton.
    A large majority of the documentation was generated using the Chat-GPT4
    LLM after code was developed to make it easier to come back to and read.

This module serves as the initializer for the `neuro_models` package, 
exposing key classes and functions for simulating and analyzing 
neuron models.

Package:
    - `neuro_models`: A package developed for the sole purpose of making my life easier
    when completing the MATH 4803 - Computational Neuroscience homeworks.

Imports:
    - ODE_Model: Abstract base class for ordinary differential equation (ODE) 
      neuron models. It provides an interface for implementing neuron models 
      that simulate voltage changes over time.
    - QIF: A concrete implementation of the ODE_Model that simulates 
      the Quadratic Integrate-and-Fire (QIF) neuron model, including spiking 
      behavior with thresholds and resets.
    - plot_voltage: A utility function for visualizing the voltage trace of 
      neuron models as a function of time, including optional spike threshold 
      and reset markers.

Public Interface:
    __all__: A list of public objects provided by this package. Only 
    'ODE_Model', 'QIF', and 'plot_voltage' are accessible for external use 
    when importing the package.

Usage:
    To use the key components from this package, you can import them directly:
    
    >>> from neuro_models.neuro_ode_models import ODE_Model, QIF
    >>> from nuero_models.utils import plot_voltage
"""

from .neuro_ode_models import ODE_Model, QIF
from .utils import plot_voltage
from .poisson_process import PoissonProcess

__all__ = ['ODE_Model', 'QIF', 'plot_voltage', 'PoissonProcess']
