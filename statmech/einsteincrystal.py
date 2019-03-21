from pylab import *
from scipy.misc import comb

import numpy as np

"""
Calculate entropy as a function of q (heat of crystal A) as two Einstein crystals come
into thermal contact with one another. The Einstein crystal has the characteristic property
T_E = epsilon/k and it predicts that the energy and heat capacities of a crystal are universal
functions of the dimensionless ratio T/T_{E}.
"""
 
NA = 300 # number of particles for gas A
NB = 200 # gas B
q = 200 # heat

multA = np.zeros(q+1,float) multB = np.zeros(q+1,float)
mult = np.zeros(q+1,float)
N = NA + NB # total number of particles

qvalue = array(range(q+1)) # output equilibrium temperatures

for ik in range(len(qvalue)):
    """
    Adjust equilibrium accordingly.
    """
    qA = qvalue[ik]
    qB = q - qA
    multA[ik] = comb(qA+NA-1,qA)
    multB[ik] = comb(qB+NB-1,qB)
    mult[ik] = multA[ik]*multB[ik]

SA = log(multA)
SB = log(multB)
STOT = SA + SB

plot(qvalue,SA,"-r",qvalue,SB,"-b",qvalue,STOT,":k")
xlabel("q_A"), ylabel("S")

"""
We can use the Monte Carlo method to simulate the time dynamics of the system
as it goes through microstates.
"""

NA = 10 # number of gas A particles
NB = 990 # number of gas B partcles
qA = 300 # A heat
qB = 9700 # B heat

q = qA + qB # Total energy
N = NA + NB # total number of particles

nstep = 1000000 # number of steps
nbetween = 1000 
state = np.zeros(N,float)

# Generate initial, random state placeA = randint(0,NA,qA)
for ip in range(len(placeA)):
    i = placeA[ip]
    state[i] = state[i] + 1
placeB = randint(0,NB,qB)+NA

for ip in range(len(placeB)):
    i = placeB[ip]
    state[i] = state[i] + 1

# Simulate state development
EA = np.zeros(nstep, float)
EB = np.zeros(nstep, float)
TBSA = np.zeros(nstep, float)
TB = np.zeros(nstep, float)

for istep in range(nstep):
    i1 = randiint(0,N) # Select oscillator at random
    if (state[i1]>0): # Check if it has energy
        i2 = randint(0,N) # Then find other oscillator state[i2] = state[i2] + 1
        state[i1] = state[i1] - 1
    # Calculate T_B S_A
    EA[istep] = sum(state[:NA-1])
    EB[istep] = sum(state[NA:])
    qA = EA[istep]
    qB = EB[istep]
    omegaA = comb(NA+qA-1,qA)
    TB[istep] = qB/NB
    TBSA[istep] = TB[istep]*log(omegaA)

"""
Plot energy and heat capacitance of an Einstein crystal
"""

x = np.linspace(0,1.5,1000)
E = 1.0/(exp(1.0/x)-1.0)
subplot(2,1,1)
plot(x,E)
xlabel("T/\theta_E")
ylabel("E/\epsilon")
subplot(2,1,2)
CV = diff(E)/diff(x)
xx = 0.5*(x[1:]+x[0:-1])
plot(xx,CV)
xlabel("T/\theta_E")
ylabel("C_V/\epsilon")
show()
