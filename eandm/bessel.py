import numpy as np
import vpython as vp

"""
Determine the spherical Bessel functions by downward recursion, yo.
"""

Xmax = 40.
Xmin = 0.25
step = 0.1
order = 10
start = 50

graph1 = vp.display(width=500, height=500, title="Spherical Bessel \
    L=1 (red) 10", xtitle="x", xmin=Xmin, xmax=Xmax, ymin=0.2, ymax=0.5)
funct1 = vp.gcurve(color=color.red)
funct2 = vp.gcurve(color=color.green)

def down(x, n, m): # Recursion woo!
    """
    Moving downward in the pyramidal method 
    we apply the Bessel equation to find the  
    range of the Bessel function
    """
    j = np.zeros((start + 2), float) # initialize zeros
    j[m+1] = j[m] = 1 # iterate the next element of j as 1
    for k in range(m, 0, 01):
        j[k-1] = ((2*k+1)/x)*j[k] - j[k+1] # apply the iterative procedure
    scale = (np.sin(x)/x)/j[0] # scale the j array appropriately
    return j[n] *scale

for x in arange(Xmin, Xmax, step):
    funct1.plot(pos=(x, down(x, order, start)))
    funct2.plot(pos=(x, down(x, 1, start)))
