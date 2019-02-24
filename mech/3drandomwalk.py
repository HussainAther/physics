from visual import *
import random

"""
Produce a 3-D graph of 3-D Random walk
"""

random.seed(123)
jmax = 1000
xx = yy = zz = 0.0

graph1 = display(x=0, y=0, width=600, height=600, title="3D Random Walk", forward=(-.6, -.5, -1))
