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

