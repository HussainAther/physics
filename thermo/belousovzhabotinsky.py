import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import convolve2d
from matplotlib import animation

"""
The Belousov-Zhabotinsky (BZ bz belousov zhabotinsky) uses non-equilibrium chemical oscillator
with periodic changes in concentration. We can create a simple reaction model using three
chemical substrates with alpha, beta, and gamma rate constants.
"""

nx, ny = 600, 450 # width and height of the image
alpha, beta, gamma = 1, 1, 1 # rate constants

def update(p,arr):
    """Update arr[p] to arr[q] by evolving in time."""

    # Count the average amount of each species in the 9 cells around each cell
    # by convolution with the 3x3 array m.
