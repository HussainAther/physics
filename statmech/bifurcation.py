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
