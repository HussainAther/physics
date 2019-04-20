import random
import numpy as np

from vpython import *

"""
Produce a 3-D graph of 3-D Random walk. It's a mathematical
formalization of a path of random steps in succession.
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
for i in range(1, 100):
    xx += (random.random() - 0.5)*2
    yy += (random.random() - 0.5)*2
    zz += (random.random() - 0.5)*2
    pts.x[i] = 200*xx - 100
    pts.y[i] = 200*yy - 100
    pts.z[i] = 200*zz - 100
    rate(100)

print("This walk’s distance R =" , np.sqrt(xx∗xx + yy∗yy+ zz∗zz))
