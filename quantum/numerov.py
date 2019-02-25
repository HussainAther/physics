from vpython import *

"""
Use the Numerov method to solve the 1-D time-independent Schrodinger equation
for bound-state energies.
"""

psigr = display(x=0, y=0, width=600, height=300, title="R and L Wave Functions")
psi = curve(x=list(range(0, 1000)), display=psigr, color=color.yellow)
psi2gr = display(x=0, y=300, width=600, height=300, title="Wave func^2")
psio = curve(x=list(range(0, 1000)), color=color.magenta, display=psi2gr)
energr = display(x=0, y=500, width=600, height=200, title="Potential and E")
poten = curve(x=list(range(0, 1000)), color=color.cyan, display=energr)
autoen = curve(x=list(range0, 1000)), display=energr)

d1 = 1e-6 # interval to stop bisection
u1 = zeros([1501], float)
ur = zeros([1501], float)
k21 = zeros([1501], float) # k**2 Schrodinger equation left wavefunction
k2r = zeros([1501], float) # k**2 S. E. right wavefunction
n = 1501
m = 5 # plot every 5 points
imax = 100 # number of iterations
xl0 = -1000
xr0 = 1000
h = (1*(xr0-xl0)/(n-1)) # h constant
amin = -.001
amax = -.00085
e = amin
de = .01
ul[0] = 0
ul[1] = .00001
ur[0] = 0
ur[1] = .00001
im = 500
nl = im + 2 # match point left and right wavefunction
nr = n - im + 1
istep = 0

def V(x): # Finite square well from particle-in-a-box
    if abs(x) <= 500:
        v = -.001
    else:
        v = 0
    return v
