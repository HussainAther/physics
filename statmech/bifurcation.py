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
