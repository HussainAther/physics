import numpy as np
import matplotlib.pyplot as plt

"""
Use the Lax-Friedrich method to solve the advection equation

U_t + vU_x = 0

U(x, t) = exp(-200*(x-xc-v*t)^2)

Similar to beamwarming.py
"""

N = 100 # number of steps
tmax = 2.5 # maximum time value
xmin = 0 # distance start
xmax = 1 # distance end
xc = .25 # curve center
v = 1 # velocity
dt = .009 # timestep

dx = (xmax - xmin)/N # step size
x = np.arange(xmin-dx, xmax+(2*dx), dx) # range of x across the step size

u0 = np.exp(-200*(x-xs)**2) # initialize energy at 0
u = u0 # same
unp1 = u0

nsteps = round(tmax/dt) # number of steps
alpha = v*dt/(2*dx) # term in our expansion

def lf():
    """
    Lax-Friedrich scheme for solving nonlinear differential equations.
    """
