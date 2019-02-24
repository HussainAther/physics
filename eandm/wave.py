from visual import *

"""
Plot the sine and cosine of the waves in 3-D
"""

xmax = 201
scene = display(x=0, y=0, width=500, height=500,
    title="sin(6pi*x/201-t)", background=(1.,1.,1.),
    forward=(-0.6,-0.5,1))

sinWave = curve(x=range(0, xmax), color=color.yellow, radius=4.5)
cosWave = curve(x=range(0, xmax), color=color.red, radius=4.5)
Xaxis = curve(pos=[(-300, 0, 0), (300, 0, 0)], color=color.blue)
