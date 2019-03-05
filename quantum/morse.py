import numpy as np
import scipy as sp

"""
Morse potential is the ptoential energy for a diatomic molecule using the interatomic
interaction model by physicist Philip Morse. It approximates the vibrational structure of
the molecule with more accuracy than the quantum harmonic oscillator by accountign for the anharmonicity
of real bonds and the non-zero transition probability for overtone and cmobination bands.
"""

def morse(l, m, u, x):
    """
    Morse potential using the equilibrium bond distance. Fit to the data using
    various variables in expanding the well potential function. This is from solving the Schrodinger
    equation with the stationary states on the Morse potential.
    l is lambda = ((2mD)/ahbar). m is mass, D is well depth (according to internuclear separation),
        a is sqrt(k_e/2D_e) in which k_e is the force constant minimum of the well, D_e is the well depth,
        hbar is planck's constant.
    m, x, and u are the coordinates that describe the distance of each atom with respect to one another.
    """
    return (l * (np.exp(-2*m*(x-u))-2*np.exp(-m*(x-u))) + v)


"""
We can use a generalized Laguerre polynomial to write the eigenstates and eigenvalues of the Morse potential.
"""

def Laguerre(alpha, n, z):

