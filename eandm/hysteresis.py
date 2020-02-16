import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint

"""
Pinched hysteresis in a memristor
"""

# Constants
eta, L, Roff, Ron, p, T, w0 = 1.0, 1.0, 70.0, 1.0, 10.0, 20.0, 0.5

t = np.arange(0.0, 40.0, 0.01)

# Set up the ODEs
def memristor(X, t):
    w = X
    dwdt = ((eta * (1 - (2*w - 1) ** (2*p)) * np.sin(2*np.pi * t/T))
           / (Roff - (Roff - Ron) * w))
    return dwdt
