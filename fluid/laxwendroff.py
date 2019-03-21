from vpython.graph import *
import numpy as np

"""
Solve the advection equation, a description of some scalar field (u) carried along by a
flow of constant speed v. Use the Lax-Wendroff scheme.
"""

# initialize parameters
m = 100
c = 1
dx = 1/m
beta = .8 # beta = c*dt/dx
u = np.zeros((m+1)/float)
u0 = np.zeros((m+1), float)
uf = np.zeros((m+1), float)
dt = beta*dx/c
T_final = .5
n = int(T_final/dt)

graph1 = gdisplay(width=600, height=500, title="Advec Eqn: Initial (red), Exact (blue), Lax-Wendroff (yellow)", xtitle="x", ytitle="u(x) Blue=exact, Yellow=Sim", xmin=0, xmax=1, ymin=0, ymax=1)
initfn = gcurve(color=color.red)
exactfn = gcurve(color=color.blue)
numfn = gcurve(color=color.yellow)

for i in range(0, m):
    x = i*dx
    u0[i] = exp(-300*(x-.12)**2)
    initfn.plot(pos=.01*i, u0[i])
    uf[i] = exp(-300*(x-.12-c*T_final)**2)
    exactfn.plot(pos=(.01*i, uf[i]))
    rate(20)

for j in range(0, n+1):
    for i in range(0, m-1):
        u[i+1] = (1-beta*beta)*u0[i+1]-(.5*beta)*(1-beta)*u0[i+2]+(.5*beta)*(1+beta)*u0[i] # Lax-Wendroff scheme
        u[0] = 0
        u[m-1] = 0
        u0[i] = u[i]

for j in range(0, m-1):
    rate(30)
    numfn.plot(pos=(.01*j, u[j]))
