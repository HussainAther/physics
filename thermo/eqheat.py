from numpy import *
import matplotlib.pylab as p
from mpl_toolkits.mplot3d import Axes3D

"""
To solve for the temperature distribution within an aluminum bar of length L = 1m that is subject to
the boudnary and initial conditions: T(x=0, t) = T(x=L, t) = 0 K and T(x, t=0) == 100K. The corresponding
thermal conductivity is 237W/mK, specific heat is 900 J/(kgK), and the aluminum density is 2700 kg/m^3.
"""

thc = 237 # thermal condivity
C = 900 # specific heat
rho = 2700 # density of Al
Nx = 101 # total x size
Nt = 3000 # total time
dx = .03 # x step size
dt = .9 # time step size
T = zeros((Nx,2), float)
Tpl = zeros((Nx, 31), float) # temperature distribution

u = [] # energies

for ix in range (1, Nx - 1):
    T[ix, 0] = 100.0  # Initial T
    T[0,0] = 0.0
    T[0,1] = 0.  # Boundary conditions

T[Nx-1, 0] = 0.
T[Nx-1, 1] = 0.0

cons = thc/(C*rho)*dt/(dx*dx) # calcuate constant
n = 1 #
for t in range (1, Nt):
    for ix in range (1, Nx - 1):
        T[ix,1] = T[ix,0] + cons * (T[ix + 1,0]+ T[ix-1,0] - 2. * T[ix,0])
        
    if t % 300 == 0 or t == 1: # Every 300 steps
        for ix in range (1, Nx - 1, 2):
            Tpl[ix, n] = T[ix, 1]
        n += 1
        
    for ix in range (1, Nx - 1):
        T[ix, 0] = T[ix, 1]

x = list(range(1, Nx - 1, 2)) # Plot alternating points
y = list(range(1, 30))
X, Y = p.meshgrid(x, y)

def f(Tpl): # function to return the temperature distribution
    z = Tpl[X, Y]
    return z

Z = f(Tpl) # collect the variables
fig = p.figure()
ax = Axes3D(fig)
ax.plot_wireframe(X, Y, Z, color = 'r')
ax.set_xlabel("Position")
ax.set_ylabel("time")
ax.set_zlabel("Temp")
p.show()
