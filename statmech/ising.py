import random
from vpython.graph import *

"""
Metropolis algorithm for a one-dimensional Ising chain.
"""

scene = display(x=0,y=0,width=700,height=200, range=40,title="Spins")
engraph = gdisplay(y=200,width=700,height=300, title="E of Spin System",\
    xtitle="iteration", ytitle="E",xmax=500, xmin=0, ymax=5, ymin=−5)
enplot = gcurve(color=color.yellow)
