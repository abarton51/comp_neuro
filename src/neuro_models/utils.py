"""
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
                 color='b', title="Voltage vs Time"):
    """Utility function to plot voltage as a function of time"""
    plt.plot(t_values, V_values, label=f'Ix = {Ix}', color=color)
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title(title)
    plt.axhline(y=yth, color='r', linestyle='--', label='V_th (Threshold)')
    plt.axhline(y=yr, color='g', linestyle='--', label='V_reset (Reset)')
    plt.legend()
    plt.grid(True)
