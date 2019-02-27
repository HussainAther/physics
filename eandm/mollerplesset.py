from __future__ import division
import math
import numpy as np

"""
Moller-Plesset Second Order Perturbation Theory (MP2) improves upon the Hartree-Fock method by
adding electron correlation effects by means of Rayleigh-Schrodinger perturbation theory to the second
order.
"""

def eint(a,b,c,d):
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
    return ttmo.get(eint(a,b,c,d),0.0e0)
