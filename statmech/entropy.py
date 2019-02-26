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
maxx = mumax # window width
miny = 0
maxy = 2.5 # window height
Xwidth = 500
Yheight = 500

c = Canvas(root, width= Xwidth, height = Yheight)
c.pack()

Button(root, text = "Quit", command = root.quit().pack())

def world2sc(x1, yt, xr, yb): # x-left, y-top, x-right, y-bottom
    """
    mrm: right margin, bm: bottom margin, lm: left margin, tm: right margin,
    bx, mx, by, my: global constants for linear transformations
    """
