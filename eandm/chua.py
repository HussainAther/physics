import numpy as np

"""
The simplest electric circuit that exhibits chaos is the Chua (chua Chua's) circuit.
It produces a waveform that never repeats itself. It's a real-world example of chaotic
behavior.
"""

def res(x):
    """
    Nonlinear resistor response to voltage x(t)
    """
    return np.sin(x) 

def ChuaEq(alpha, beta, x, y, z):
    """
    For x(t), y(t), and z(t) voltages across capacitors C1 and C2 and current in
    inductor L1, respectively, we describe three equations to illustrate the 
    dynamics of Chua's circuit.
    """
    dxdt = alpha*(y - x -res(x))
