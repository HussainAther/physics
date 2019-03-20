import numpy as np

"""
We use the Beam-Warming method for solutions of the linear advection equation (used to
describe the transport of a substance or quantity by bulk motion.

First we use the Navier-Stokes equatino for incompressible flow. We use
an implicit Beam-Warming scheme for the non-linear hyperbolic equation.
We can create a conservative form of this equation to linearize it.
Then we add a dissipation term for non-linear hyperbolic equations (if
we have a shock wave), and use a second-order smoothing term
if we only require a stable solution. Then we solve the resulting equation
using the Thomas Algorithm (modified tridiagonal matrix algorithm).
"""
