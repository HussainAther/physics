import numpy as np

"""

"""

def morse(q, m, u, x ):
    """
    Morse potential using the equilibrium bond distance
    """
    return (q * (np.exp(-2*m*(x-u))-2*np.exp(-m*(x-u))))
