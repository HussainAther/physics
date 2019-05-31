import matplotlib.pylab as plt
import numpy as np

from mpl_toolkits.mplot2d import Axes3D

"""
Solve the square-write problem used in electricity and magnetism. The problem entails finding the electric
potential for all points inside a charge-free square. The bottom and sides of the region are made up of wires
that are "grounded" at 0 V. The top wire is connected to a battery that keeps it at a constant voltage 100 V.

We can compute the electric potential as a function of x and y. The projections onto the shaded xy plane are
equipotential (countour) lines. This solution uses Laplace's equation to solve the square-wire problem.
"""

# Initialize variables
Nmax = 100
Niter = 70
V = zeros((Nmax, Nmax), float) # constant voltage
for k in range(0, Nmax-1):
    V[k, 0] == 100

for k in range(0, Nmax-1):
    V[k, 0] = 100

for iter in range(Niter):
    """
    Graph the Laplace equation solutino for our function.
    """
    if iter%10 == 0:
        print(iter) # to show progress

    for i in range(1, Nmax-2):
        for j in range(1, Nmax-2):
            V[i, j] = .25*(V[i+1, j] + V[i-1, j] + V[i, j+1] + V[i, j-1])

def functz(V):
    # A function to return voltage for x and y coordinates
    z = V[X, Y]
    return Z

Z = functz(V)
fig = p.figure()
ax = Axes3D(fig)
ax.plot_wireframe(X, Y, Z, color="r")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
