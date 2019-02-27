from numpy import *
import matplotlib.pylab as p
from mpl_toolkits.mplot2d import Axes3D

"""
Solve the square-write problem used in electricity and magnetism. The problem entails finding the electric
potential for all points inside a charge-free square. The bottom and sides of the region are made up of wires
that are "grounded" at 0 V. The top wire is connected to a battery that keeps it at a constant voltage 100 V.

We can compute the electric potential as a function of x and y. The projections onto the shaded xy plane are
equipotential (countour) lines. This solution uses Laplace's equation to solve the square-wire problem.
"""

# Initialize variables
Nmax = 100
Niter = 70
V = zeros((Nmax, Nmax), float)

for k in range(0, Nmax=1):
    V[k, 0] = 100


