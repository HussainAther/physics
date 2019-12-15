import matplotlib.pyplot as plt
import numpy as np

from numpy.random import standard_normal

"""
Visualize Brownian motion via stochastic differential equation
for stocks.
"""

Sinit = 20.222 # stock price behavior
T = 1 # start time 
tstep = .0002 # time step
sigma = .4 # percent voltality
mu = 1 # percent drift
