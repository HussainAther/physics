import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint

"""
Pinched hysteresis in a memristor
"""

# Constants
eta, L, Roff, Ron, p, T, w0 = 1.0, 1.0, 70.0, 1.0, 10.0, 20.0, 0.5
