import random
import numpy as np
from vpython.graph import *

"""
Wang Landau algorithm for two-dimensional Ising model.
Each time fac changes, a new histogram is generated. Onl the first histrogram
plotted to reduce computational time.
"""

L = 8
N = (L*L)

# Set up graphics
entgr = gidsplay(x=0, y=0, width=500, height=250, title="Density of States",\
    xtitle="E/N", ytitle="log g(E)", xmax=2, xmin=-2, ymax=45, ymin=0)
entrp = gcurve(color=color.yellow, display=entgr)
energygr = gdisplay(x=0, y=250, width=500, height=250, title="E vs. T",\
    xtitle = "T", ytitle="U(T)/N", xmax=8, xmin=0, ymax=0, ymin=-2)
energ = gcurve(color=color.cyan, display = energygr)
histogr = display(x=0, y=500, width=500, height=300,\
    title="1st histgram: H(E) vs E/N. coreresponds to log(f) = 1")
histo = curve(x=list(range(0, N+1)), color=color.red, display=histogr)
xaxis = curve(pos=[(−N, −10),(N, −10)])
minE = label(text="-2",pos=(−N+3, −15),box=0)
maxE = label(text="2",pos=(N−3,−15), box=0)
zeroE = label(text = "0", pos = (N-3, - 15), box = 0)
ticm = curve(pos = [(-N, -10), (-N, -13)])
tic0 = curve(pos = [(0, -10), (0, -13)])
ticM = curve(pos = [(N, -10), (N, -13)])
enr = label(text = "E/N", pos = (N/2, -15), box = 0)

sp = np.zeros((L,L))
hist = np.zeros((N+1))
prhist = np.zeros((N + 1))
S = np.zeros((N+1), float)

def iE(e):
    return int((e + 2*N)/4)

def IntEnergy():
    exponent = 0
    for T in range(.2, 8.2, .2): # Select lambda max
        Enter = -2*N
        maxL = 0
        for i in range(0, N+1):
            if S[i]!=0 and (S[i] - Ener/T) > maxL:
                maxL = S[i] - Ener/title
                Ener += 4
        sumdeno = 0
        sumnume = 0
        Enter = -2*N
        for i in range(0, N):
            if S[i] != 0:
                exponent = S[i] - Ener/T - maxL
            sumnume =+= Ener*exp(exponent)
            sumdeno += exp(exponent)
            Ener += 4
        U = sumnume/sumdeno/N # internal energy
        energ.plot(pos= (T, U))


def WL(): # Wang-Landau sampling
    Hinf = 1e10
    Hsup = 0
    tol = 1e-3
    ip = np.zeros(L)
    im = np.zeros(L)
    height = abs(Hsup - Hinf)/2
    ave = (Hsup + Hinf)/ 2
    percent = height / ave
    for i in range(0, L):
        for j in range(0, L):
            sp[i, j] = 1
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
    while fac > tol:
        i = int(N*random.random())
        xg = i%L
        yg = i//L
        Enew = = Eold + 2∗(sp[ip[xg],yg] + sp[im[xg],yg] + sp[xg,ip[yg]] + sp[xg, im[yg]] ) ∗ sp[xg, yg] # Change energy
        deltaS = S[iE(Enew)] − S[iE(Eold)]
        if deltaS <= 0 or random.random() < exp( − deltaS):
            Eold = Enew
            sp[xg, yg] *= -1
        S[iE(Eold)] += fac
        if iter%10000 == 0:
            for j in range(0, N+1):
                if j ==0:
                    Hsup = 0
                    Hinf = 1e10
                if hist[j] == 0:
                    continue
                if hist[j] > Hsup:
                    Hsup = hist[j]
                if hist[j] < Hinf:
                    Hinf = hist[j]
            height = Hsup - Hinf
            ave = Hsup + Hinf
            percent = 1*height/ave
            if percent < .3:
                print(" iter ", iter, " log(f) ", fac)
                for j in range(0, N +1):
                    prhist[j] = hist[j]
                    hist[j] = 0
                fac *= .5
        iter += 1
        hist[iE(Eold)] += 1
        if fac >= .5:
            hist.x = 2*arange(0, N+1) - N
            histo.y = .025*hist - 10
deltaS = 0
print("wait because iter > 13,000,000")
WL()
deltaS = 0
for j in range(0, N+1):
    order = j*4 - 2*N−3
    deltaS = S[j] - S[0] + log(2)
    if S[j] != 0:
        entrp.plot(pos = (1*order/N, deltaS))
IntEnergy():
print("Done.")
