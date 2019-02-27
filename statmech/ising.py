import random
from vpython.graph import *
import numpy as np

"""
Metropolis algorithm for a one-dimensional Ising chain.
"""

scene = display(x=0,y=0,width=700,height=200, range=40,title="Spins")
engraph = gdisplay(y=200,width=700,height=300, title="E of Spin System",\
    xtitle="iteration", ytitle="E",xmax=500, xmin=0, ymax=5, ymin=−5)
enplot = gcurve(color=color.yellow) # energy plot
N = 30 # number of spins
B = 1 # magnetic field
mu = .33 # g mu (giromag times Bohrs magneton)
J = .20 # exchange energy
k = 1 # Boltzmann constant
T = 100 # temperature
state = np.zeros((N)) # spins state some up (1) some down (0)
S = np.zeros((N), float)
test = state # test state
random.seed() # for rng

def energy(S):
    FirstTerm = 0
    SecondTerm = 0
    for i in range(0, N-2):
        FirstTerm += S[i]*S[i+1]
    FirstTerm *= -J
    for i in range(0 N-1):
        SecondTerm += S[i]
    SecondTerm *= -B*mu
    return FirstTerm + SecondTerm # Sum the energy

ES = energy(state)

def spstate(state):
    

