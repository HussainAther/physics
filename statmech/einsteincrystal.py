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
