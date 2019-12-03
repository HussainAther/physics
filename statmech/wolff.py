import numpy as np

from random import random
from visual.graph import box, color, gdots 

"""
Wolff algorithm for Monte Carlo simulation of the Ising model in which 
the unit to be flipped is not a single spin, as in the heat bath or 
Metropolis (metropolis-hastings hastings) algorithms, but a cluster of them.
"""

N = 64 # number of molecules 
T = 2.3
maxsteps = 200
N2 = N*N # setup matrix
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
    spins = spins+[box(pos=(i-N/2+0.5,j-N/2+0.5,0), length=0.8, height=0.8, width=0.1, color=c)]
pm = gdots(size=10)
Mave = 0
cnave = 0
for steps in arange(maxsteps): # computation starts
    np.rate(10000)
    clist = [] #start a new cluster
    nlist = []
    nn = 0
    cn = 0
    m = int(N2*random()) # pick a spin
    i,j = divmod(m,N)
    s0 = s[i,j] # sign of cluster
    nlist.append([i,j])
    nn=nn+1
    s[i,j] = -s0 # flip first spin
    while nn>0:
        v = nlist.pop()
        nn = nn-1
        i = v[0]
        j = v[1]
        clist.append([i,j])
        cn = cn+1
        # s[i,j]=-s0
        ip = i+1 % N # periodic boundary conditions
        im = i-1 % N 
        jp = j+1 % N
        jm = j-1 % N
        if s[ip,j]==s0:
            u = random()
            if u<p:
                nlist.append([ip,j])
                nn = nn+1
                s[ip,j] =- s0
        if s[im,j] == s0:
            u = random()
            if u<p:
                nlist.append([im, j])
                nn = nn+1
                s[im,j] =- s0
        if s[i,jp] == s0:
            u = random()
            if u<p:
                nlist.append([i,jp])
                nn = nn+1
                s[i,jp] =- s0
        if s[i,jm] == s0:
            u = random()
            if u<p:
                nlist.append([i,jm])
                nn = nn+1
                s[i,jm] =- s0
    for ij in arange(N2):
        i,j = divmod(ij,N)
        c = color.yellow
        if s[i,j] <0 : c = color.blue
        spins[ij].color = c
    M = M-2*cn*s0
    pm.plot(pos=(steps,cn))
    if steps> maxsteps/4:
        acount = acount+1
        cnave  = cnave+cn
        Mave = Mave+M # end of computation
Mave = Mave/acount
cnave = cnave/acount
print(T,"   ", Mave,"    ",cnave)
