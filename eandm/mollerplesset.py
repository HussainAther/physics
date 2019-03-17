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

def teimo(a,b,c,d):
    """
    Return Value of spatial MO two electron integral
    Example: (12\vert 34) = tei(1,2,3,4)
    """
    return tt.get(compound(a,b,c,d),0.0e0)

"""
Initialize the orbital energies and transformed two-electron integrals
"""
Nelec = 2 # we have 2 electrons in HeH+
dim = 2 # we have two spatial basis functions in STO-3G
E = [-1.52378656, -0.26763148]
tt = {5.0: 0.94542695583037617, 12.0: 0.17535895381500544, 14.0: 0.12682234020148653, 17.0: 0.59855327701641903, 19.0: -0.056821143621433257, 20.0: 0.74715464784363106}

"""
Convert them from spatial coordinates to coordinates that flow with spin molecular orbital theory.
"""

dim *= 2
ints = np.zeros((dim, dim, dim, dim))

