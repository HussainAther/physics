import numpy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

"""
Poisson's equation is obtained from adding a source term to the right-hand-side of Laplace's equation:

∂2p/∂x2+∂2p/∂y2 = b
So, unlinke the Laplace equation, there is some finite value inside the field that affects the solution.
Poisson's equation acts to "relax" the initial sources in the field.
"""
