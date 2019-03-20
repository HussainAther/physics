import numpy as np

"""
We use the Beam-Warming method for solutions of the linear advection equation (used to
describe the transport of a substance or quantity by bulk motion.

We solve the advection equation

U_t + vU_x = 0

over the spatial domain 0 <= x <= 1 discretized into nodes with dx = .01 using the Beam-Warming
scheme given by an initial profile of a Gaussian curve:

U(x,t) = exp(-200*(x-xc-v*t)^2)

in which xc  is the center of the curve at t=0.

First we use the Navier-Stokes equatino for incompressible flow. We use
an implicit Beam-Warming scheme for the non-linear hyperbolic equation.
We can create a conservative form of this equation to linearize it.
Then we add a dissipation term for non-linear hyperbolic equations (if
we have a shock wave), and use a second-order smoothing term
if we only require a stable solution. Then we solve the resulting equation
using the Thomas Algorithm (modified tridiagonal matrix algorithm).


"""

N = 100 # number of steps
tmax = 2.5 # maximum time value
xmin = 0 # distance start
xmax = 1 # distance end
xc = .25 # curve center
v = 1 # velocity

dx = (xmax - xmin)/N # step size
x = np.arange(xmin-dx, xmax+(2*dx), dx) # range of x across the step size

u0 = np.exp(-200*(x-xs)**2) # initialize energy at 0
u = u0 # same
unp1 = u0

nsteps = round(tmax/dt) # number of steps
alpha1 = v*dt/(2*dx) # first guesses
alpha2 = v**2*dt**2/(2*dx**2) 
