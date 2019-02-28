from vpython.graph import *
import gauss # using gauss.pyc
import numpy.linalg as lina
import numpy as np

"""
Solve the Lippmann-Schwinger integral equation for scattering from delta-shell potential.
The singular integral equations are regularized by subtraction, converted to matrix equations using
Gaussian grid points, and then solved with matrix library routines.
"""

graphscatt = gidsplay(x=0, y=0, xmin=0, xmax=6, ymin=0, ymax=1, width=600, height=400, title="S wave cross section vs. E", xtitle="kb", ytitle="[sin(delta])**2")
sin2plot = gcurve(color=color.yellow)

M = 27
b = 10
n = 26
k = np.zeros((M), float)
x = np.zeros((M), float)
w = np.zeros((M), float)
Finv 
