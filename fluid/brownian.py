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

# standard colors
colors = [(214, 27, 31), 
          (148, 104, 189),
          (229, 109, 0),
          (41, 127, 214),
          (217, 119, 194),
          (44, 160, 44),
          (227, 119, 194),
          (72, 17, 121),
          (196, 156, 148)]

# Scale the colors to a [0, 1] range
for i in range(len(colors)):
    r, g, b = colors[i]
    colors[i] = (r/255, g/255, b/255)

# Plot.
plt.figure(figsize=(12,12))

Steps = round(T/tstep) # steps in years
S = np.zeros([NumSimulation, Steps], dtype=float)
x = range(0, int(Steps))

for j in range(0, NumSimulation):
    S[j,0] = Sinit
