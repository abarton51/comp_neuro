"""
Author: Austin Barton
Collaborators: None
Credits: **ALL** code was written, developed, and used solely by Austin Barton.
    A large majority of the documentation was generated using the Chat-GPT4
    LLM after code was developed to make it easier to come back to and read.

This module contains utility functions for plotting and analyzing voltage as a 
function of time in computational neuroscience models, such as the Quadratic 
Integrate-and-Fire (QIF) model.

Functions:
----------
- plot_voltage(t_values, V_values, Ix=None, yth=None, yr=None, color='b', title="Voltage vs Time"):
    Utility function to plot voltage over time with optional parameters for 
    injected current (Ix), threshold voltage (V_th), and reset voltage (V_reset).
"""

import matplotlib.pyplot as plt

def plot_voltage(t_values, V_values, Ix=None, yth=None, yr=None,
                 color='b', title="Voltage vs Time", label_name="Ix", figsize=None):
    """Utility function to plot voltage as a function of time"""
    if figsize: plt.figure(figsize=figsize)
    plt.plot(t_values, V_values, label=f'{label_name} = {Ix}', color=color)
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title(title)
    if yth:
        plt.axhline(y=yth, color='r', linestyle='--', label='V_th (Threshold)')
    if yr:
        plt.axhline(y=yr, color='g', linestyle='--', label='V_reset (Reset)')
    plt.legend()
    plt.grid(True)
