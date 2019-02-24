from numpy.linalg import solve
from vpython.graph import *

"""
Use the Newton-Raphson search to solve the two-mass-on-a-string problem
"""

scene = display(x=0, y=0, width=500, height=500, title="String and masses configuration")

tempe = curve(x=range(0, 500), color=color.black)
