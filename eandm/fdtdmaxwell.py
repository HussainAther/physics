from vpython import *

"""
Finite-Difference Time Domain Method of findings solutions to Maxwell's equations for linearly
polarized wave propogation in the z-direction in free space.
"""

xmax = 201
ymax = 100
zmax = 100

scene = display(x=0, y=0, width=800, height=500, title="E: cyan, H: red. Periodic Boundary conditions",
            forward=(-.6,-.5,-1))
Efield = curve(x=list(range(0, xmax), color=color.cyan, radius=1.5, display=scene)
Hfield = curve(x=list(range(0, xmax), color=color.cyan, radius=1.5, display=scene)
vplane = curve(pos=[(-xmax, ymax), (xmax, ymax), (xmax, -ymax), (-xmax, -ymax),
    (-xmax, ymax)], color=color.cyan)
