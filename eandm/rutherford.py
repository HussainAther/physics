from math import sqrt,log,cos,sin,pi
from random import random

"""
Rutherford scattering is the elastic scattering of charged particles by the Coulomb interaction.
It is a physical phenomenon explained by Ernest Rutherford in 1911 that led to the development of
the planetary Rutherford model of the atom and eventually the Bohr model.
"""

# Constants
Z = 79 # atomic number
e = 1.602e-19 # electric charge for electron
E = 7.7e6*e # electric field
epsilon0 = 8.854e-12 # permittivity of free space
a0 = 5.292e-11 # radius
sigma = a0/100 # coulomb potential
N = 1000000 # number of particles
