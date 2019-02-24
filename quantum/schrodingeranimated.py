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
PlotObj = curve(x=xs, color=color.yellow, radius=0.1)
g.center=(0,2,0)

psr = exp(-.5*(xs/.5)**2) * cos(k0*xs)
psi = exp(-.5*(xs/.5)**2) * sin(k0*xs)
v = 15.0*xs**2

while True:
    rate(500)
    psr[1:-1] = psr[1:-1] - (dt/dx2)*(psi[2:]+psi[:-2]-2*psi[1:-1]) + dt*v[1:-1]*psi[1:-1]
    psi[1:-1] = psi[1:-1] - (dt/dx2)*(psr[2:]+psr[:-2]-2*psr[1:-1]) + dt*v[1:-1]*psr[1:-1]
    PlotObj.y = 4*(psr**2 + psi**2)
