from __future__ import division

import math
import numpy as np

"""
Moller-Plesset Second Order Perturbation Theory (MP2) improves upon the Hartree-Fock method by
adding electron correlation effects by means of Rayleigh-Schrodinger perturbation theory to the second
order.
"""

def compound(a,b,c,d):
    """
    Return compound index given four indices
    """
    if a > b: ab = a*(a+1)/2 + b
    else: ab = b*(b+1)/2 + a
    if c > d: cd = c*(c+1)/2 + d
    else: cd = d*(d+1)/2 + c
    if ab > cd: abcd = ab*(ab+1)/2 + cd
    else: abcd = cd*(cd+1)/2 + ab
    return abcd

def sm2e(a,b,c,d):
    """
    Return Value of spatial MO two electron integral
    Example: (12\vert 34) = tei(1,2,3,4)
    """
    return tee.get(compound(a,b,c,d),0.0e0)

"""
Initialize the orbital energies and transformed two-electron integrals
"""
Nelec = 2 # two electrons in HeH+
dim = 2 # two spatial basis functions in STO-3G
E = [-1.52378656, -0.26763148] # two electron energies
tee = {5.0: 0.94542695583037617, 12.0: 0.17535895381500544, 14.0: 0.12682234020148653, 17.0: 0.59855327701641903, 19.0: -0.056821143621433257, 20.0: 0.74715464784363106} # two-electron energy combinations

"""
Convert them from spatial coordinates to spin molecular orbital theory.
"""

dim *= 2
ints = np.zeros((dim, dim, dim, dim)) 
for p in range(1,dim+1):  
    for q in range(1,dim+1):  
        for r in range(1,dim+1):  
            for s in range(1,dim+1):  
                val1 = sm2e((p+1)//2,(r+1)//2,(q+1)//2,(s+1)//2) * (p%2 == r%2) * (q%2 == s%2)  
                val2 = sm2e((p+1)//2,(s+1)//2,(q+1)//2,(r+1)//2) * (p%2 == s%2) * (q%2 == r%2)  
                ints[p-1,q-1,r-1,s-1] = val1 - val2

"""
Spin basis fock matrix eigenvalues
"""

fs = np.zeros((dim))  
for i in range(0,dim):  
    fs[i] = E[i//2]  

"""
Electron Moller Plesset second order calculation.
"""

EMP2 = 0.0  
for i in range(0,Nelec):  
    for j in range(0,Nelec):  
        for a in range(Nelec,dim):  
            for b in range(Nelec,dim):  
            	EMP2 += 0.25*ints[i,j,a,b]*ints[i,j,a,b]/(fs[i] +fs[j] -fs[a] - fs[b])
