from visual.graph import *

"""
Determine the spherical Bessel functions by downward recursion, yo.
"""

Xmax = 40.
Xin = 0.25
step = 0.1
order = 10
start = 50
graph1 = gdisplay(width=500, height=500, title="Spherical Bessel",
    L=1 (red), 10', xtitle="x", title="j(x)", xmin=Xmin, xmax=Xmax, ymin=0.2, ymax=0.5)
funct1 = gcurve(color=color.red)
