from vpython.graph import *

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
funct2 = gcurve(color=color.green)
def down(x, n, m): # Recursion woo!
    j = zeros((start + 2), float)
    j[m+1] = j[m] = 1
    for k in range(m, 0, 01):
        j[k-1] = ((2*k+1)/x)*j[k] - j[k+1]
    scale = (sim(x)/x)/j[0]
    return j[n] *scale
    
for x in range(Xmin, Xmax, step):
    funct1.plot(pos=(x, down(x, order, start)))
    funct2.plot(pos=(x, down(x,1,start)))
