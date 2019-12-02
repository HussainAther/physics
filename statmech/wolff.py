import numpy as np

from __future__ import division, print_function
from random import random
from visual import divmod, 
from visual.graph import box, color   

"""
Wolff algorithm for Monte Carlo simulation of the Ising model in which the unit to be flipped is not a single spin, as in the heat bath or Metropolis (metropolis-hastings hastings) algorithms, but a cluster of them.
"""

N = 64 #inputs
T = 2.3
maxsteps = 200
N2 = N*N #setup
print("maxsteps=", maxsteps, "T=", T)
s = ones( (N,N), dtype=int )
acount = 0
M = N2
print("N=", N, "M(0) =", M)
p = 1-np.exp(-2/T)
spins = []  # graphics
for ij in np.arange(N2):
    i,j = divmod(ij,N)
    c = color.yellow
    spins=spins+[box(pos=(i-N/2+0.5,j-N/2+0.5,0), length=0.8, height=0.8, width=0.1, color=c)]
