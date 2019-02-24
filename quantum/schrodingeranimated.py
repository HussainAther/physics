from visual import *

"""
Animated visualizations of the Schrodinger's equation solutions
"""

# wave function, probability, potential
dx = 0.04
dx2 = dx**2
k0 = 5.5*pi
dt = dx2/20.0
xmax = 6.0

g = display(width=500, height=250, title="Wave packet in harmonic oscillator potential")
PlotObj = curve
