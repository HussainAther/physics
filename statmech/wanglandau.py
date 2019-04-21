#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import random
import numpy as np

from vpython.graph import *

"""
Wang Landau (Wang-Landau) algorithm for two-dimensional Ising model.
Each time fac changes, a new histogram is generated. Only the first histrogram
plotted to reduce computational time.
"""

L = 8 # size of our lattice array
N = (L*L) # make it two-dimensional

# graph
entgr = gidsplay(x=0, y=0, width=500, height=250, title="Density of States", xtitle="E/N", ytitle="log g(E)", xmax=2, xmin=-2, ymax=45, ymin=0)
entrp = gcurve(color=color.yellow, display=entgr)
energygr = gdisplay(x=0, y=250, width=500, height=250, title="E vs. T", xtitle = "T", ytitle="U(T)/N", xmax=8, xmin=0, ymax=0, ymin=-2)
energ = gcurve(color=color.cyan, display = energygr)
histogr = display(x=0, y=500, width=500, height=300, title="1st histgram: H(E) vs E/N. coreresponds to log(f) = 1")
histo = curve(x=list(range(0, N+1)), color=color.red, display=histogr)
xaxis = curve(pos=[(−N, −10),(N, −10)])
minE = label(text="-2",pos=(−N+3, −15),box=0)
maxE = label(text="2",pos=(N−3,−15), box=0)
zeroE = label(text = "0", pos = (N-3, - 15), box = 0)
ticm = curve(pos = [(-N, -10), (-N, -13)])
tic0 = curve(pos = [(0, -10), (0, -13)])
ticM = curve(pos = [(N, -10), (N, -13)])
enr = label(text = "E/N", pos = (N/2, -15), box = 0)

# initialize values
sp = np.zeros((L,L)) # spins
hist = np.zeros((N+1)) # histogram values for results
prhist = np.zeros((N + 1)) # second histogram for step in middle
S = np.zeros((N+1), float) # entropy

def iE(e):
    """
    Normalize the energy state.
    """
    return int((e + 2*N)/4)

def IntEnergy():
    """
    Initialize with energy of two-dimensional Ising Lattice
    """
    exponent = 0
    for T in np.arange(.2, 8.2, .2): # Select lambda max
        Enter = -2*N
        maxL = 0
        for i in range(0, N+1):
            if S[i]!=0 and (S[i] - Ener/T) > maxL:
                maxL = S[i] - Ener/T 
                Ener += 4
        sumdeno = 0
        sumnume = 0
        Enter = -2*N
        for i in range(0, N):
            if S[i] != 0:
                exponent = S[i] - Ener/T - maxL # calculate exponential value
            sumnume += Ener*exp(exponent) # add to our sum
            sumdeno += exp(exponent)
            Ener += 4
        U = sumnume/sumdeno/N # internal energy
        energ.plot(pos= (T, U))


def WL(): 
    """
    Wang-Landau sampling
    """
    Hinf = 1e10 # for histogram
    Hsup = 0 
    epsilon = 1e-3 # tolerance
    ip = np.zeros(L) 
    im = np.zeros(L) # BC R or down, L or up
    height = abs(Hsup - Hinf)/2 # initialize histogram
    ave = (Hsup + Hinf)/ 2 # average # about average of histogram
    percent = height / ave # normalize with respect to our average 
    for i in range(0, L):
        for j in range(0, L):
            sp[i, j] = 1 # begin spins
    for i in range(0, L):
        ip[i] = i + 1
        im[i] = i - 1
    ip[L-1] = 0
    im[0] = L -1
    Eold = -2*N
    for j in range(0, N+1):
        S[j] = 0
    iter = 0
    fac = 1
    while fac > epsilon:
        i = int(N*random.random()) # select a random spin
        xg = i%L # remainder for spin values
        yg = i//L # localize x, y, grid point
	# cost function
        Enew = Eold + 2∗(sp[ip[xg],yg] + sp[im[xg],yg] + sp[xg,ip[yg]] + sp[xg, im[yg]] ) ∗ sp[xg, yg] # change energy
        deltaS = S[iE(Enew)] − S[iE(Eold)] # change entropy
        if deltaS <= 0 or random.random() < exp( − deltaS): 
            Eold = Enew
            sp[xg, yg] *= -1 # flip spin
        S[iE(Eold)] += fac # change entropy again
        if iter%10000 == 0: # flatness every 10000 sweeps
            for j in range(0, N+1): 
                if j ==0: # for our first iteration
                    Hsup = 0
                    Hinf = 1e10
                if hist[j] == 0: # do nothing
                    continue
                if hist[j] > Hsup: # cost function
                    Hsup = hist[j]
                if hist[j] < Hinf:
                    Hinf = hist[j]
            height = Hsup - Hinf # update our values
            ave = Hsup + Hinf
            percent = height/ave
            if percent < .3: # is the histogram flat?
                print(" iter ", iter, " log(f) ", fac) # tell us
                for j in range(0, N +1): 
                    prhist[j] = hist[j] 
                    hist[j] = 0
                fac *= .5
        iter += 1
        hist[iE(Eold)] += 1 
        if fac >= .5: # only the first histogram
            hist.x = 2 * np.arange(0, N+1) - N
            histo.y = .025*hist - 10

deltaS = 0
print("wait because iter > 13,000,000")
WL() # run the Wang-Landau algorithm

deltaS = 0
for j in range(0, N+1):
    order = j*4 - 2*N−3
    deltaS = S[j] - S[0] + log(2)
    if S[j] != 0:
        entrp.plot(pos = (order/N, deltaS)) # plot the entropy

IntEnergy():
print("Done.")
