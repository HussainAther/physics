from vpython import *

"""
Solve the time-dependent Schrödinger equation for a particle
described by a Gaussian wave packet moving within a harmonic oscillator potential.
"""

# initialize wave function, probability, and potential
dx = 0.04,
dx2 = dx*dx
k0 = 5.5*pi
dt = dx/20
xmax = 6
xs = arange(-xmax, xmax+dx/2,dx)

g= display(
