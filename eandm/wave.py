import numpy as np
import vpython as vp

"""
Plot the sine and cosine of the waves in 3-D
"""

xmax = 201
scene = vp.display(x=0, y=0, width=500, height=500,
    title="sin(6pi*x/201-t)", background=(1.,1.,1.),
    forward=(-0.6,-0.5,1))

sinWave = vp.curve(x=range(0, xmax), color=color.yellow, radius=4.5)
cosWave = vp.curve(x=range(0, xmax), color=color.red, radius=4.5)
Xaxis = vp.curve(pos=[(-300, 0, 0), (300, 0, 0)], color=color.blue)

incr = np.pi/xmax # x increment
for t in range(0, 10, .02): # time loop
    for i in range(0, xmax): # loop through x values
        x = i*incr
        f = np.sin(6.0*x-t)
        zz = np.cos(6.0*x-t)
        yp = 100*f
        xp = 200*x-300
        zp = 100*zz
        sinWave.x[i] = xp
        sinWave.y[i] = yp
        cosWave.x[i] = xp
        cosWave.z[i] = zp
