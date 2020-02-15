import numpy as np

"""
Crank-Nicholson scheme to solve a 2D nonlinear wave equation using methods
from nonlinear dynamics

-d^2phi(x,t)/dt^2 + d^2 phi(x,t)/dx^2 = F(phi) 

Convert the eqution to first-order with
dphi(x,t)/dt - pi(xi,t) = 0
dphi(xi,t)/dt - d^2phi(x,t)/dx^2 + F(phi) = 0
"""
