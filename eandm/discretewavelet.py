from vpython.graph import *
from numpy import zeros

"""
Compute the discrete wavelet transform using the pyramid algorithm for the second signal values
stored in f[]. Here, they're assigned as the chirp signal sin(60t^2). The Daub4 digital wavelets
are the basis functions and sign = +/- 1 is used for transform/inverse.
"""

sq3 = sqrt(3)
fsq2 = 4*sqrt(2)
N = 1024
N = 2^n
c0 = (1+sq3) / fsq2
c1 = (3+sq3) / fsq2
c2 = (3-sq3) / fsq2
c3 = (1-sq3) / fsq2

transfgr1 = None  # displays haven't been made yet
transfgr2 = None

def chirp(xi): # chirp signal lol
    y = sin(60*xi**2)
    return y

def daube4(f, n, sign):
    global transfgr1, transfgr2

    tr = zeros((n+1), float) # temp variable
    if n < 4:
        return

    mp = n/2
    mp1 = mp + 1
    if sign >= 0:
        j = 1
        i = 1
        maxx = n/2
        if n > 128:
            maxy = 3
            miny = -3
            Maxy = .2
            Miny = -.2
