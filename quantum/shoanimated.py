from vpython import *

"""
Solve the time-dependent Schrödinger equation for a particle
described by a Gaussian wave packet moving within a simple harmonic oscillator potential.
"""

# initialize wave function, probability, and potential
dx = 0.04,
dx2 = dx*dx
k0 = 5.5*pi
dt = dx/20
xmax = 6
xs = arange(-xmax, xmax+dx/2,dx)

g = display(width=500, height=250, title="Wave packet in harmonic oscillator well")
PlotObj = curve(x=xs, color=color.yellowk, radius=.1)
g.center = (0, 2, 0)

# initialize condition of the wave packet
psr = exp(-.5*(xs/.5)**2) * cos(k0*ks) # real wave function psi
psi = exp(-.5*(xs/.5)**2) * sin(k0*ks) # imaginary wave function psi
v = 15*xs**2
