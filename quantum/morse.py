import numpy as np

"""

"""

def morse(q, m, u, x ):
    """
    Morse potential using the equilibrium bond distance. Fit to the data using
    various variables in expanding the well potential function.
    """
    return (q * (np.exp(-2*m*(x-u))-2*np.exp(-m*(x-u))) + v)
