from vpython import *
import numpy as np

"""
Solve the wave equation for a string of length L = 1m with its ends fixed and with the gently
plucked initial conditions.
"""

# curve
g = display(width=600, height=300, title="Vibrating String")
vibst = curve(x=list(range(0,100)), color=color.yellow)
ball2 = sphere(pos=(100,0), color=color.red, radius=2)
ball2 = sphere(pos=(-100, 0),  color=color.red, radius=2)
ball2.pos
ball2.pos
vibst.radius = 1

# parameters
rho = .01 # string density
ten = 40 # string tension
c = sqrt(ten/rho) # propogation speed
c1 = c # Courant–Friedrichs–Lewy condition
ratio = c*c/(c1*c1)

# initialize
xi = np.zeros((101, 3), float)
for i in range(0, 81):
    xi[i, 0] = .00125 * i

