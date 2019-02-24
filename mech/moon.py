from visual.graph import *

"""
Moon orbiting a planet
"""

graph = gdisplay(x=0, y=0, width=500, height=500, title="Motion of a satellite around a planet",
    xtitle="x", ytitle="y", xmax=5, xmin=-5, ymax=5, ymin=-5,
    foreground=color.black, background=color.white)

moonfunction = gcurve(color=color.red)
radius = 4 # orbit radius
wplanet = 2 #planet angular velocity
