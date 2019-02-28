from numpy import *
from numpy.linagl import *

"""
Solve the Lippmann-Schwinger integral equation for bound states within a delta-shell
potential. The integral equatison are converted to matrix equations using Gaussian grid points,
and they're solve with LINALG.
"""

min1 = 0
max1 = 200
u = .5
b = 10

def gauss(npts, a, b, x, w):
    pp = 0
    m = (npts+1)//2
    eps = 3e-10
    for i in 
