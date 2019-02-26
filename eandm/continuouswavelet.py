import matplotlib.pylab as p:
from mpl.toolkits.mplot2d import Axes3D
from vpython import *

"""
Calculates a normalized continuous wavelet transform of the signal data in "inpu"
using Morlet wavelets. The discrete wavelet transform is faster and yields a compressed transform,
but is less transparent
"""

invtrgr = display(x=0, y=0, width=600, height=200, title="Inverse TF")
invtr = curve(x=list(range(0,240)), display=invtrgr , color=color.green)

iT = 0.0
noPtsSig = N
iS= 0.1
fT = 12.0
noS = 20
tau= iTau
W = fT − iT
N = 240
noTau = 90
iTau = 0.
s= iS

# Need ∗very∗ small s steps for high frequency if s small
dTau = W/noTau
dS = (W/iS)∗∗(1./noS)
maxY = 0.001
sig = zeros (( noPtsSig ) , float )

def signal(noPtsSig , y):
    t = 0.0
    hs = W/noPtsSig
    t1 = W/6.
    t2 = 4.∗W/6.
    for i in range (0 , noPtsSig ):
        if t >= iT and t <= t1:
            y[i] = sin(2∗pi∗t)
    elif t >= t1 and t <= t2:
        y[i] = 5.∗sin(2∗pi∗t)+10.∗sin(4∗pi∗t)
    elif t >= t2 and t <= fT:
        y[i] = 2.5∗sin(2∗pi∗t) + 6.∗sin(4∗pi∗t) + 10.∗sin(6∗pi∗t)
    else:
        print("In signal(...) : t out of range.")
        sys . exit (1)
    t += hs

signal(noPtsSig, sig)
Yn = zeros((noS+!, noTau+1), float)

