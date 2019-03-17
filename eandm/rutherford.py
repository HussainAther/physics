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

def gaussian():
    """
    Generate two random Gaussian numbers for our distribution.
    """
    r = sqrt(-2*sigma*sigma*log(1-random()))
    theta = 2*pi*random()
    x = r*cos(theta)
    y = r*sin(theta)
    return x,y

# simulate the transmissions
count = 0
for i in range(N):
    x,y = gaussian()
    b = sqrt(x*x+y*y)
    if b<Z*e*e/(2*pi*epsilon0*E):
        count += 1

print(count,"particles were reflected out of",N)
