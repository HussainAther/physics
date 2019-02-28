from vpython import *
import numpy as np

"""
Solve Maxwell's equations with FDTD time-stepping for circularly polarized wave propogations
in the z-direction in free space.
"""

scene = display(x=0, y=0, width=600, height=400, range=200, title="Circular polarization: E field \
    in white. H field in yellow")
global phy, pyx
max = 201

c = .01 # Courant stability condition

# set arrays
Ex = np.zeros((max+2, 2), float)
Hy = np.zeros((max+2, 2), float)
Ey = np.zeros((max+2, 2), float)
Hx = np.zeros((max+2, 2), float)

arrowcol = color.white
Earrows = []
Harrows = []

for i in range(0, max, 10):
    Earrows.append(arrow(pos=(0, i-100), axis=(0,0,0), color=arrowcol))
    Harrows.append(arrow(pos=(0, i-100), axis=(0,0,0), color=color.yellow))
