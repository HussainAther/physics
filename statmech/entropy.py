try:
    from tkinter import *
except:
    from Tkinter import *
import math
from numpy import zeros

"""
Calculate Shannon enetropy for the logistic map as a function of growth parameter mu.
"""

global Xwidth, Yheight

Tk( ): root.title("Entropy versus mu ")
mumin = 3.5
mumax = 4
dmu = .005
nbin = 1000
nmax = 100000
prob = zeros((1000), float)
minx = mumin
maxx = mumax
miny = 0
maxy = 2.5
Xwidth = 500
Yheight = 500
