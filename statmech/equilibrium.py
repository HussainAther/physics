from pylab import *

import matplotlib.pyplot as plt
import numpy as np

"""
Simple method of adjusting distances to achieve equilibrium.
"""

n = rand(20)
N = 20

for i in range(1,nstep):
    r = rand(1) # create a single random value 
    if (r < n[i-1]/N): # if we need to adjust distances
        n[i] = n[i-1] - 1 # Move atom from left to right else:
        n[i] = n[i-1] + 1 # Move atom from right to left
        plt(range(0,nstep),n/N, xlabel="t", ylabel="n/N")
        plt.show()
