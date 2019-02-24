from visual import *
import random

"""
Produce a 3-D graph of 3-D Random walk
"""

random.seed(123)
jmax = 1000
xx = yy = zz = 0.0

graph1 = display(x=0, y=0, width=600, height=600, title="3D Random Walk", forward=(-.6, -.5, -1))

# Create the curve
pts = curve(x=list(range(0, 100)), radius=10.0, color=color.yellow)
xax = curve(x=list(range(0, 1500)), color=color.red, pos=[(0, 0, 0), (1500, 0, 0)], radius=10.)
yax = curve(x=list(range(0, 1500)), color=color.red, pos=[(0, 0, 0), (0, 1500, 0)], radius=10.)
zax = curve(x=list(range(0, 1500)), color=color.red, pos=[(0, 0, 0), (0, 0, 1500)], radius=10.)
xname = label(text="X", pos=(1000, 150, 0), box=0)
yname = label(text="Y", pos=(-100, 1000, 0), box=0)
zname = label(text="Z", pos=(100, 0, 1000), box=0)

# Starting point
pts.x[0] = pts.y[0] = pts.z[0]

