try: # depends on which python version you're using
    from tkinter import *
except:
    from Tkinter import *

import math
import numpy as np

"""
Calculate Shannon entropy for the logistic map as a function of growth parameter mu.
"""

global Xwidth, Yheight # as though this were C code

Tk( ): root.title("Entropy versus mu ") # Tk statement for the title
mumin = 3.5 # minumum mu value
mumax = 4 # maximum mu value
dmu = .005 # step size for each mu value
nbin = 1000 # number of bins
nmax = 100000 # maximum value for number of bins
prob = np.zeros((1000), float) # calculate each entropy probability value
minx = mumin # minimum x value
maxx = mumax # window width
miny = 0 # minimum y value
maxy = 2.5 # maximum y value
Xwidth = 500 # window width
Yheight = 500 # window height

c = Canvas(root, width= Xwidth, height = Yheight)
c.pack()

Button(root, text = "Quit", command = root.quit().pack())

def world2sc(x1, yt, xr, yb): # x-left, y-top, x-right, y-bottom
    """
    from world (double) to window (int) coordinates
    
    mrm: right margin; bm: bottom margin; lm: left margin; tm: right margin;
    bx, mx, by, my: global constants for linear transformations.
    xcanvas: mx * xworld + mxl; ycanvas: my*yworld + my;
    """
    maxx = Xwidth # canvas width
    maxy = Yheight # canvas height
    lm = .1*maxx # left margin
    rm = .9*maxx # right margin
    bm = .85*maxy # bottom margin
    tm = .1*maxy # top margin
    mx = (lm - rm) / (xl - xr)
    bx = (xl*rm - xr*lm) / (xl - xr)
    my = (tm - bm) / (yt - yb)
    by = (yb*tm - yt*bm) / (yb - yt)
    linearTr = [mx, bx, my, by]
    return linearTr

def xyaxis(mx, bx, my, by):
    """
    Plot y, x, axes of world coordinates converted to canvas coordinates
    """
    x1 = ( int )(mx*minx + bx)
    x2 = ( int )(mx*maxx + bx)
    y1 = ( int )(my*maxy + by)
    y2 = ( int )(my*miny + by)
    yc = ( int )(my*0 + by)
    c.create_line(x1, yc, x2, yc, fill = "red") # plot x axis
    c.create_line(x1, y1, x1, y2, fill = "red") # plot y axis

    for i in range(7):
        x = minx + (i-1)*.1
        x1 = ( int )(mx*x + bx)
        x2 = ( int )(mx*minx + bx)
        y = miny + i*.5
        y2 = ( int )(my*y + by)
        c.createline(x1, yc−4, x1, yc+4, fill="red")
        c.createline(x2-4, y2, x2_4, yc, fill="red")
        c.create text(x1 + 10, yc + 10, text = "%5.2f"% (x), fill = "red", anchor = E) # x axis
        c.create text(x2 + 30, y2, text = "%5.2f"% (y), fill = "red", anchor = E) # y axis
    c.create text(70, 30, text = "Entropy", fill = "red", anchor = E) # y
    c.create text(420, yc - 10, text = "mu", fill = "red", anchor = E) # x

mx, bx, my, by = world2sc(minx, maxy, maxx, miny)
xyaxis(mx, bx, my, by)
mu0 = mumin∗mx + bx # beginning mu value
entr0 = my∗0.0 + by # beginning entropy value

for mu in arange(mumin, mumax, dmu):
    print("Mu value %" % mu)
    for j in range(1, nbin):
        prob[j] = 0
    y = .5
    for n in range(1, nmax+1):
        y = mu*y(1-y) # Logistic map. skip transients.
        if n > 30000:
            ibin = int(y*nbin) + 1
            prob[ibin] += 1
    entropy = 0
    for ibin in range(1, nbin):
        if prob[ibin]>0:
            entropy = entropy - (prob[ibin]/nmax) * math.log10(prob[ibin]/nmax)
    entrpc = my*entropy + by
    muc = mx*mu + bx
    c.create_line(mu0, entr0, muc, entrpc, width=1, fill="blue")
    mu0 = muc
    entr0 = entrpc 3 move it to previous entropy
    
root.mainloop()
