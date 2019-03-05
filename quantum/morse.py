import numpy as np

"""
Morse potential is the ptoential energy for a diatomic molecule using the interatomic
interaction model by physicist Philip Morse. It approximates the vibrational structure of
the molecule with more accuracy than the quantum harmonic oscillator by accountign for the anharmonicity
of real bonds and the non-zero transition probability for overtone and cmobination bands.
"""

def morse(q, m, u, x ):
    """
    Morse potential using the equilibrium bond distance. Fit to the data using
    various variables in expanding the well potential function.
    """
    return (q * (np.exp(-2*m*(x-u))-2*np.exp(-m*(x-u))) + v)
