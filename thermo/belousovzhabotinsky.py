import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import convolve2d
from matplotlib import animation

"""
The Belousov-Zhabotinsky (BZ bz belousov zhabotinsky) uses non-equilibrium chemical oscillator
with periodic changes in concentration. We can create a simple reaction model using three
chemical substrates with alpha, beta, and gamma rate constants.
"""
