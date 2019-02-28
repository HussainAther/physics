from vpython.graph import *
import numpy as np

"""
Solve the advection equation, a description of some scalar field (u) carried along by a
flow of constant speed v. Use the Lax-Wendroff scheme.
"""

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

