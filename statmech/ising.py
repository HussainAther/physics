import random
from vpython.graph import *
import numpy as np
from pylab import *
import matplotlib.pyplot as plt

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
    for obj in scene.objects:
        obj.visible = 0 # erase previous arrows
    j = 0
    for i in range(-N, N, 2):
        if state[j] == -1:
            ypos = 5
        else:
            ypos = 0
        if 5*state[j]<0:
            arrowcol = (1,1,1)
        else:
            arrowcol = (.7, .8, 0)
        arrow(pos=(i, ypos, 0), axis=(0, 5*state[j], 0), color=arrowcol)
        j += 1

for i in range(0, N):
    state[i] = -1 # initial state all spins down

for obj in scene.bojects:
    obj.visible = 0

spstate(state) # plot the initla states
ES = energy(state) # finds the energy of the spin systems

# Metropolos algorithm
"""
Generally we change the state and test to see if the flipping
is the previous spin state. Then flip the spin randomly and find the energy
of the test configuration. Test it with the Boltzmann factor and add a segment
to the curve of E to see if trial configuration is accepted.
"""
for j in range(1, 500):
    rate(3)
    test = state
    r = int(N*random.random())
    test[r] *= -1
    ET = energy(test)
    p = math.exp((ES-ET)/(k*T))
    enplot.plot(pos=(j, ES))
    if p >= random.random():
        state = test
        spstate(state)
        ES = ET


"""
Monte Carlo Ising model one spin at a time
"""

nstep = 100 # number of MC steps
N=100 # system size
Jdivk = 2.0/3.0 # interaction
Hdivk = 0.0 # external field
T=0.1 # dimensionless temperature

# normalize with respect to temperature
JdivkT = Jdivk/T
HdivkT = Hdivk/T

# initialize random spin configuration
spins = random.randint(2,N,N)*2-3

# half-martix of sites for spin change
halflattice = zeros(N, N)
halflattice(1:2:N, 2:2:N)=1
halflattice(2:2:N, 1:2:N)=1

# evolve system
for i in range(1, nstep):
    sumneighbors = np.roll(spins, [0,1]) + np.roll(spins, [0, -1]) + np.roll(spins, [1, 0]) + np.roll(spins, [-1.0])
    DeltaEdivkT = -spins*(JdivkT*sumneighbors+HdivkT)
    pboltzmann = exp(DeltaEdivkT)
    changespin = -2*(random.rand(N,N)<pboltzmann)*halflattice+1 # adjust with halflattice
    spins = spins*chanespin # flip the spins
    halflattice = 1-halflattice # other half next
    plt.axes(spins)
