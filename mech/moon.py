import numpy as np

from visual.graph import *

"""
Equations of motion for a moon orbiting a planet.
"""

graph = gdisplay(x=0, y=0, width=500, height=500, title="Motion of a satellite around a planet",
    xtitle="x", ytitle="y", xmax=5, xmin=-5, ymax=5, ymin=-5,
    foreground=color.black, background=color.white)

moonfunction = gcurve(color=color.red) # use gcurve
planetradius = 4 # planet orbit radius
wplanet = 2 # planet angular velocity
moonradius = 1 # moon radius around planet
wmoon = 14 # moon angular velocity around planet
for time in np.arange(0, 3.2, .02): # iterate
    rate(20)
    x = planetradius*cos(wplanet*time) + moonradius*cos(wmoon*time)
    y = planetradius*sin(wplanet*time) + moonradius*sin(wmoon*time)
    moonfunction.plot(pos=(x,y,))
