from pylab import *
from scipy.misc import comb

"""
Calculate entropy as a function of q (heat of crystal A) as two Einstein crystals come
into thermal contact with one another. The Einstein crystal has the characteristic property
T_E = epsilon/k and it predicts that the energy and heat capacities of a crystal are universal
functions of the dimensionless ratio T/T_{E}.
"""
 
NA = 300
NB = 200
q = 200 # heat

multA = zeros(q+1,float) multB = zeros(q+1,float)
mult = zeros(q+1,float)
N = NA + NB

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

NA = 10
NB = 990
qA = 300
qB = 9700

q = qA + qB # Total energy
N = NA + NB

nstep = 1000000
nbetween = 1000
state = zeros(N,float)

# Generate initial, random state placeA = randint(0,NA,qA)
for ip in range(len(placeA)):
    i = placeA[ip]
    state[i] = state[i] + 1
placeB = randint(0,NB,qB)+NA

for ip in range(len(placeB)):
    i = placeB[ip]
    state[i] = state[i] + 1

# Simulate state development
EA = zeros(nstep,float)
EB = zeros(nstep,float)
TBSA = zeros(nstep,float)
TB = zeros(nstep,float)

for istep in range(nstep):
    i1 = randiint(0,N) # Select oscillator at random
    if (state[i1]>0): # Check if it has energy
        i2 = randint(0,N) # Then find other oscillator state[i2] = state[i2] + 1
        state[i1] = state[i1] - 1
    # Calculate T_B S_A
    EA[istep] = sum(state[:NA-1])
    EB[istep] = sum(state[NA:])
