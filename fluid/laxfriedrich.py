import numpy as np
import matplotlib.pyplot as plt

"""
Use the Lax-Friedrich method to solve the advection equation. It uses a Forward
Euler method to discretize the system of ordinary differential equations.

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
    tc = 0
    for i in range(nsteps):
        plt.clf()
        for j in range(N+2): # Lax-Friedrich scheme
            unp1[j] = u[j] - alpha*(u[j+1] - u[j-1]) + (1/2)*(u[j+1] - 2*u[j] + u[j-1])
        u = unp1
        # periodic boundary conditions
        u[0] = u[N+1]
        u[N+2] = u[1]

        uexact = np.exp(-200*(x-xc-v*tc)**2) # exact energy solution for comparison with lf

        plt.plot(x, uexact, "r", label="Exact solution")
        plt.plot(x, u, "bo-", label="Lax-Friedrich")
        plt.axis((xmin-0.15, xmax+0.15, -0.2, 1.4))
        plt.grid(True)
        plt.xlabel("Distance (x)")
        plt.ylabel("u")
        plt.legend(loc=1, fontsize=12)
        plt.suptitle("Time = %1.3f" % (tc+dt))
        plt.pause(0.01)
        tc += dt
