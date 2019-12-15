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

colors = [(214, 27, 31), 
          (148, 104, 189),
          (229, 109, 0),
          (41, 127, 214),
          (217, 119, 194),
          (44, 160, 44),
          (227, 119, 194),
          (72, 17, 121),
          (196, 156, 148)]
