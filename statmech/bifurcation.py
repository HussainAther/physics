import matplotlib.pyplot as plt
import numpy as np

"""
Bifurcations are transitions between dynamical states used in nonlinear dynamics.
"""

def xeq1(r):
    """
    Stable equilibrium
    """
    return np.sqrt(r)

def xeq2(r):
    """
    Unstable equilibrium
    """
    return np.sqrt(r)

# Plot.
fig = plt.figure(figsize=(9,6))
ax1 = fig.add_subplot(1, 1, 1)
domain = linspace(0, 10)
ax1.plot(domain, xeq1(domain), "b-", label = "stable equilibrium", linewidth = 3)
ax1.plot(domain, xeq2(domain), "r--", label = "unstable equilibrium", linewidth = 3)
ax1.legend(loc="upper left")
#neutral equilibrium point
ax1.plot([0], [0], "go")
ax1.axis([-10, 10, -5, 5])
ax1.set_xlabel("r")
ax1.set_ylabel("x_eq")
ax1.set_title("Saddle-node bifurcation")
