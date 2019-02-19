import numpy
from matplotlib import pyplot
import time, sys

"""

The one-dimensional diffusion equation is:

∂u/∂t = ν (∂2u/∂x2)

This equation has a second-order derivative that we will discretize. The second-order derivative can be represented geometrically
as the line tangent to the curve given by the first derivative. We will discretize the second-order
derivative with a Central Difference scheme: a combination of Forward Difference and Backward Difference of the first
derivative. Consider the Taylor expansion of ui+1 and ui−1 around ui:


ui+1=ui+Δx∂u∂x∣∣∣i+Δx22∂2u∂x2∣∣∣i+Δx33!∂3u∂x3∣∣∣i+O(Δx4)
ui−1=ui−Δx∂u∂x∣∣∣i+Δx22∂2u∂x2∣∣∣i−Δx33!∂3u∂x3∣∣∣i+O(Δx4)

If we add these two expansions, you can see that the odd-numbered derivative terms will cancel each other out.
If we neglect any terms of O(Δx4) or higher (and really, those are very small), then we can rearrange the sum of these
two expansions to solve for our second-derivative.

ui+1+ui−1=2ui+Δx2∂2u∂x2∣∣∣i+O(Δx4)

Then rearrange to solve for (∂2/u∂x2)∣∣∣i and the result is:

∂2u/∂x2=ui+1−2ui+ui−1Δx2+O(Δx2)
"""

nx = 42
dx = 2 / (nx - 1)
nt = 20    # number of timesteps
nu = 0.3   # viscosity value
sigma = .2
dt = sigma * dx**2 / nu

"""
We can now write the discretized version of the diffusion equation in 1D:

un+1i−uniΔt=νuni+1−2uni+uni−1Δx2
As before, we notice that once we have an initial condition, the only unknown is un+1i, so we
re-arrange the equation solving for our unknown:

un+1i=uni+νΔtΔx2(uni+1−2uni+uni−1)
The above discrete equation allows us to write a program to advance a solution in time. But we need
an initial condition. Let's continue using our favorite: the hat function. So, at t=0, u=2 in the
interval 0.5≤x≤1 and u=1 everywhere else. We are ready to number-crunch!
"""


u = numpy.ones(nx)
u[int(.5 / dx):int(1 / dx + 1)] = 2

un = numpy.ones(nx)

for n in range(nt):
    un = u.copy() 
    for i in range(1, nx - 1):
        u[i] = un[i] + nu * dt / dx**2 * (un[i+1] - 2 * un[i] + un[i-1])

pyplot.plot(numpy.linspace(0, 2, nx), u);
