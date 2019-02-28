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

graph1 = gdisplay(width=600, height=500, title="Advec Eqn: Initial (red), Exact (blue), Lax-Wendroff (yellow)",
                xtitle="x", ytitle="u(x) Blue=exact, Yellow=Sim", xmin=0, xmax=1, ymin=0, ymax=1)
initfn = gcurve(color=color.red)
exactfn = gcurve(color=color.blue)
