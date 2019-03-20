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

N = 100 # number of nodes
tmax = 2.5 # maximum time value
xmin = 0 # distance start
xmax = 1 # distance end
xc = .25 # curve centre
