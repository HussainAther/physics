from vpython import *
import math

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

incr = math.pi/xmax # x increment
for t in range(0, 10, .02): # time loop
    for i in range(0, xmax): # loop through x values
        x = i*incr
        f = math.sin(6.0*x-t)
        zz = math.cos(6.0*x-t)
        yp = 100*f
        xp = 200*x-300
        zp = 100*zz
        sinWave.x[i] = xp
        sinWave.y[i] = yp
        cosWave.x[i] = xp
        cosWave.z[i] = zp
