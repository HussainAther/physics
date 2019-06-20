import numpy as np
import matplotlib.pyplot as plt

"""
We use the Beam-Warming (beam warming beamwarming) method for solutions of the linear advection equation (used to
describe the transport of a substance or quantity by bulk motion. This method is second-order
accurate.

We solve the advection equation

U_t + vU_x = 0

over the spatial domain 0 <= x <= 1 discretized into nodes with dx = .01 using the Beam-Warming
scheme given by an initial profile of a Gaussian curve:

U(x,t) = exp(-200*(x-xc-v*t)^2)

in which xc  is the center of the curve at t=0.

First we use the Navier-Stokes equatino for incompressible flow. We use
an implicit Beam-Warming scheme for the nonlinear hyperbolic equation.
We can create a conservative form of this equation to linearize it.
Then we add a dissipation term for nonlinear hyperbolic equations (if
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
diss = np.exp(-200*(x-xs)**2)[2:-2] # dissipative term

nsteps = round(tmax/dt) # number of steps
alpha1 = v*dt/(2*dx) # terms in our expansion
alpha2 = v**2*dt**2/(2*dx**2)

def bw(d=True):
    """
    Beam-Warming scheme for solving nonlinear differential equations.
    If d = True, we are under a shock wave and add a dissipative/dissipation term.
    """
    tc = 0
    for i in range(nsteps):
        plt.clf()
        for j in range(N+3): # apply the Beam-Warming scheme by using the Thomas algorithm
                           # (tridiagonal matrix algorithm) in solving the linear system of equations that result
                           # from the trapzeoidal formula (the Taylor expansion of the first term).
            unp1[j] = u[j] - alpha1*(3*u[j] - 4*u[j-1] + u[j-2]) + alpha2*(u[j] - 2*u[j-1] + u[j-2])
            if j <= N - 2 and j >= 2:
                diss[j] = u[j+2] - 4*u[j+1] + 6*u[j] - 4*u[j-1] + u[j-2]
        
        u = unp1
        # periodic boundary conditions
        u[0] = u[N+2]
        u[1] = u[N+1]
        uexact = np.exp(-200*(x - xc - v*tc)**2) # exact energy value for comparison with bw

        plt.plot(x, uexact, "r", label="Exact solution")
        plt.plot(x, u, "bo-", label="Beam-Warming")
        plt.axis((xmin-0.15, xmax+0.15, -0.2, 1.4))
        plt.grid(True)
        plt.xlabel("Distance (x)")
        plt.ylabel("u")
        plt.legend(loc=1, fontsize=12)
        plt.suptitle("Time = %1.3f" % (tc+dt))
        plt.pause(0.01)
        tc += dt
