from vpython.graph import *
import gauss # using gauss.pyc
import numpy.linalg as lina

"""
Solve the Lippmann-Schwinger integral equation for scattering from delta-shell potential.
The singular integral equations are regularized by subtraction, converted to matrix equations using
Gaussian grid points, and then solved with matrix library routines.
"""
