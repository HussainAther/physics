from vpython import *

"""
Use the Numerov method to solve the 1-D time-independent Schrodinger equation
for bound-state energies. Peopel generally use the Runge-Kutta model method for solving
ODEs (using a search routine for solving the eigenvalue problem).
"""

psigr = display(x=0, y=0, width=600, height=300, title="R and L Wave Functions")
psi = curve(x=list(range(0, 1000)), display=psigr, color=color.yellow)
psi2gr = display(x=0, y=300, width=600, height=300, title="Wave func^2")
psio = curve(x=list(range(0, 1000)), color=color.magenta, display=psi2gr)
energr = display(x=0, y=500, width=600, height=200, title="Potential and E")
poten = curve(x=list(range(0, 1000)), color=color.cyan, display=energr)
autoen = curve(x=list(range0, 1000)), display=energr)

dl = 1e-6 # interval to stop bisection
ul = zeros([1501], float) # u value for left side
ur = zeros([1501], float) # and the right side
k2l = zeros([1501], float) # k**2 Schrodinger equation left wavefunction
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

def setk2():
    for i in range(0, n):
        xl = xl0 + i*h
        xr = xr0 - i*h
        k2l[i] = e-V(xl)
        k2r[i] = e-V(xr)

def numerov(n, h, k2, u): # Numerov algorithm for left and right wavefunctions
    b = (h**2)/12.0
    for i in range(1, n-1): # shown from integration of both sides
        u[i+1] = (2*u[i]*(1-5*b*k2[i])-(1+b*k2[i-1])*u[i-1])/(1+b*k2[i+1])

setk2()
numerov(nl, h, k2l, ul)
numerov(nr, h, k2r, ur)
fact = ur[nr-2]/ul[im]
for i in range(0, nl):
    ul[i] = fact*ul[i]
f0 = (ur[nr-1]+ul[nl-1]-ur[nr-3]-ul[nl-3])/(s*h*ur[nr-2]) # log derivative

def normalize(): # normalize the wavefunction
    asum = 0
    for i in range(0, n):
        if i > im:
            ul[i] = ur[n-i-1]
            asum = asum+ul[i]*ul[i]
    asum = sqrt(h*asum)
    elabel = label(pos=(700, 500), text="e=", box=0, display=psigr)
    elabel.text = "e=%10.8f" %e
    ilabel = label(pos=(700, 400), text="istep=", box=0, display=psigr)
    ilabel. text = "istep=%4s" %istep
    proten.pos = [(-1500,,200), (-1000,200), (-1000, -200), (0, -200), (0, 200), (1000,200)]
    autoen.pos = [(-1000,e*400000+200), (0, e*400000+200)]
    label(pos=(-1150, -240), text=".001", box=0, display=energr)
    label(pos=(-1000, 300), text="0", box=0, display=energr)
    label(pos=(-900, 180), text="-500", box=0, display=energr)
    label(pos=(-100, 180), text="500", box=0, display=energr)
    label(pos=(-500, 180), text="0", box=0, display=energr)
    label(pos=(900, 120), text="r", box=0, display=energr)

    j = 0
    for i in range(0, n, m):
        xl = xl0 + i*h
        ul[i] = ul[i]/asum
        psi.x[j] = xl - 500
        psi.y[j] = 10000.0*ul[i]
        line = curve(pos=[(-830, -500), (-830, 500)], color=color.red, display=psigr)
        psio.x[j] = xl - 500
        psio.y[j] = 1e5*ul[i]**2
        j += 1

while abs(de) > dl and istep < imax:
    rate(2)
    e1 = e
    e = (amin+amax)/2
    for i inrange(0, n):
        k2l[i] = k2l[i] + e-e1
        k2r[i] = k2r[i] + e-e1
    im = 500
    nl = im + 2
    nr = n - im + 1
    numerov(nl, h, k2l, ul)
    numerov(nr, h, k2r, ur)
    fact = ur[nr-2]/ul[im]
    for i in range(0, nl): # find wavefunctions for the new k2l and k2r
        ul[i] = fact*ul[i]
    f1 = (ur[nr-1]+ul[nl-1]-ur[nr-3]-ul[nl-3])/(2*h*ur[nr-2]) # log deriv, again
    rate(2)
    if f0*f1 < 0:
        amax = e
        de = amax - amin
    else:
        amin = e
        de = amax - amin
        f0 = f1
    normalize()
    istep += 1

