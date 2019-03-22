import numpy as np
import time, sys

from matplotlib import pyplot
"""
The 1-D Linear Convection equation is the simplest, most basic model that can be used to learn
something about CFD. It is surprising that this little equation can teach us so much! Here it is:

∂u∂t+c∂u∂x=0
With given initial conditions (understood as a wave), the equation represents the propagation of
that initial wave with speed c, without change of shape. Let the initial condition be u(x,0)=u0(x).
Then the exact solution of the equation is u(x,t)=u0(x−ct).

We discretize this equation in both space and time, using the Forward Difference scheme for the time
derivative and the Backward Difference scheme for the space derivative. Consider discretizing the
spatial coordinate x into points that we index from i=0 to N, and stepping in discrete time intervals
of size Δt.

From the definition of a derivative (and simply removing the limit), we know that:

∂u∂x≈u(x+Δx)−u(x)Δx
Our discrete equation, then, is:

un+1i−uniΔt+cuni−uni−1Δx=0
Where n and n+1 are two consecutive steps in time, while i−1 and i are two neighboring points of the
discretized x coordinate. If there are given initial conditions, then the only unknown in this discretization
is un+1i. We can solve for our unknown to get an equation that allows us to advance in time, as follows:

un+1i=uni−cΔtΔx(uni−uni−1)
"""

nx = 42 # obligatory meaning of life
dx = 2 / (nx-1)
nt = 25    # number of timesteps
dt = .025  # amount of time for each step
c = 1      # assume wavespeed of c = 1

u = np.ones(nx)
u[int(.5 / dx):int(1 / dx + 1)] = 2

"""
For every element of our array u, we need to perform the operation un+1i=uni−cΔtΔx(uni−uni−1)
We'll store the result in a new (temporary) array un, which will be the solution u for the next
time-step. We will repeat this operation for as many time-steps as we specify and then we can see how
far the wave has convected.

We first initialize our placeholder array un to hold the values we calculate for the n+1 timestep,
using once again the NumPy function ones().
"""

un = np.ones(nx)

for n in range(nt):
    un = u.copy()
    for i in range(1, nx):
        u[i] = un[i] - c * dt / dx * (un[i] - un[i-1])

pyplot.plot(np.linspace(0, 2, nx), u);
