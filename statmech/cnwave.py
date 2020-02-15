import numpy as np

"""
Crank-Nicholson scheme to solve a 2D nonlinear wave equation using methods
from nonlinear dynamics

-d^2phi(x,t)/dt^2 + d^2 phi(x,t)/dx^2 = F(phi) 

Convert the eqution to first-order with
dphi(x,t)/dt - pi(xi,t) = 0
dphi(xi,t)/dt - d^2phi(x,t)/dx^2 + F(phi) = 0
"""

class cnwaveeq:
    """
    Methods for the conversion shown above.
    """
    def __init__(self, F, dF, args, xmin, xmax, Nx):
       """
       F(phi, args) = is a user-specified function returning F(phi) 
       dF(phi, args) = is a user-specified function returning dF/dphi(phi)
       xmin, xmax define the spatial integration domain
       Nx = is the number of points, so dx=(xmax-xmin)/(N-1)
       All initial data set to zero, t set to 0.
       """
