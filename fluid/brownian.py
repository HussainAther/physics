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
NumSimulation = 6
colors = ["b", "g", "r", "c", "m", "k"]

# Plot.
plt.figure(figsize=(12,12))

Steps = int(round(T/tstep)) # steps in years
S = np.zeros([NumSimulation, Steps])
x = range(0, int(Steps))

for j in range(0, NumSimulation):
    S[j,0] = Sinit
    for i in x[:-1]:
        S[j, i+1] = S[j,i]+S[j,i]*(mu-.5*pow(sigma, 2))*tstep+sigma*S[j,i]*np.sqrt(tstep)*standard_normal()

    plt.plot(x, S[j], linewidth=2, color=colors[j])

plt.title("%d Brownian motion simulations using %d Steps, \n$\sigma$=%.6f $\mu$=%.6f$S_O$=%.6f" % (int(NumSimulation), int(Steps), sigma, mu, Sinit), fontsize=18)
plt.xlabel("Steps", fontsize=16)
plt.grid(True)
plt.ylabel("Stock price", fontsize=16)
plt.ylim(0, 90)
plt.show()
