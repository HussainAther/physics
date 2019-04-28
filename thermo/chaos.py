import numpy as np

"""
We observe chaos when we split separatrices of Hamiltonian systems in homo-
and heteroclinic orbits.
"""

def eq(t, x):
    """
    An equation of motion for which we will find the coresponding Hamiltonian
    vector field.
    """

dx2d2t + (omega**2 + epsilon * np.cos(t) * np.sin(x) = 0 
