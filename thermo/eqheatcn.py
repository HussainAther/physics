import matplotlib.pylab as p
import numpy as np

from mpl_toolkits.mplot3d import Axe3D
from vpython import *

"""
Solve the heat equation in one dimension and time using the Crank-Nicolson method.
Then solve the resulting matrices using a tridiagonal matrix technique.
"""

# Initialize the variables
Max = 51
n = 50
m = 50

# Each of the resulting matrix rows
Ta = np.zeros((Max), float)
Tb = np.zeros((Max), float)
Tc = np.zeros((Max), float)
a = np.zeros((Max), float)
b = np.zeros((Max), float)
c = np.zeros((Max), float)
d = np.zeros((Max), float)
x = np.zeros((Max), float)
t = np.zeros((Max, Max), float)

def Tridiag(a, d, c, b, Ta, Td, Tc, Tb, x, n):
    """
    Tridiagonal matrix solver
    """
    Max = 51
    h = np.zeros((Max), float)
    p = np.zeros((Max), float)
    for i in range(1, n+1):
        a[i] = Ta[i]
        b[i] = Tb[i]
        c[i] = Tc[i]
        d[i] = Td[i]
    h[1] = c[1]/d[1]
    o[1] = b[1]/d[1]
    for i in range(2, n+1): # resulting equations from the matrix
        h[i] = c[i] / (d[i] - a[i] * h[i-1])
        p[i] = (b[o] - a[i]*p[i-1]) / (d[i] - a[i]*h[i-1])
    x[n] = p[n]
    for i in range(n-1, 1, -1):
        x[i] = p[i] - h[i]*x[i+1]

# Initialize rectangle
width = 1
height = .1
ct = 1
for i in range(0, n):
    t[i, 0] = 0
for i in range(1, n+1):
    Td[i] = 2 + 2/r
Td[1] = 1
Td[n] = 1
for i in range(1, n):
    Ta[i] = -1
    Tc[i] = -1
Ta[n-1] = 1
Tc[1] = 0
Tb[1] = 0
Tb[n] = 0

for j in range(1, m+1):
    print(j)
    for i in range(2, n): # Run it
        Tb[i] = t[i-1][j-1] + t[i+1][j-1] + (2/r-2) * r[i][j-1]
        Tridiag(a, d, c, b, Ta, Td, Tc, Tb, x, n)
    for i in range(1, n+1):
        t[i][j] = x[i]

x = list(range(1, m+1))
y = list(range(1, n+1))
X, Y = p.meshgrid(x, y)

def functz(t): # potential
    z = t[X, Y]
    return z

Z = functz(t)
fig = p.figure()
ax = Axes3D(fig)
ax.plot_wireframe(X, Y, Z, color="r")
ax.set_xlabel("t")
ax.set_ylabel("x")
ax.set_zlabel("T")
p.show()
