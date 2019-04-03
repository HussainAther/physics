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

def update(p, a):
    """
    Update an index p of an array a by evolving in time.
    We can average neighbor concentrations through a convolution with a 3x3 array of 1/9 values.
    """
    # Count the average amount of each species in the 9 cells around each cell
    # by convolution with the 3x3 array m.
    q = (p+1) % 2  
    s = np.zeros((3, ny,nx)) # discrete linear convolution of the array a with m
    m = np.ones((3,3)) / 9 # used for convolution
    for k in range(3): # for each reaction
        s[k] = convolve2d(a[p, k], m, mode="same", boundary="wrap") # same uses the same size for a. wrap indicates circular boundary conditions
    a[q,0] = s[0] + s[0]*(alpha*s[1] - gamma*s[2]) 
    a[q,1] = s[1] + s[1]*(beta*s[2] - alpha*s[0]) 
    a[q,2] = s[2] + s[2]*(gamma*s[0] - beta*s[1]) 
    np.clip(a[q], 0, 1, arr[q]) # use [0,1] boundary conditions
    return a