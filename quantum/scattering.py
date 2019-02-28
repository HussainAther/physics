from vpython.graph import *
from gauss import gauss
import numpy.linalg as lina

"""
Solev the Lippmann-Schwinger integral equation for scattering from delta-shell potential.
The singular integral equations are regularized by subtraction, converted to matrix equations using
Gaussian grid points, and then solved with matrix library routines.
"""
