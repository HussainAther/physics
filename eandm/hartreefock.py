from __future__ import division
import sys
import math
import numpy as np
from numpy import genfromtxt
import csv


"""
Tthe Hartree–Fock (HF) method is a method of approximation for the determination of the
wave function and the energy of a quantum many-body systemin a stationary state. It's a self-consistent
field method. It's generally the powerhouse of computational chemistry and molecular orbital theory. It's the
first step before performing more advanced calculations such as Møller-Plesset second-order Perturbation theory.

HF assumes that the exact, N-body wave function of the system can be approximated by a single
Slater determinant (in the case where the particles are fermions) or by a single permanent
(in the case of bosons) of N spin-orbitals. By invoking the variational method, one can derive a
set of N-coupled equations for the N spin orbitals. A solution of these equations yields the
Hartree–Fock wave function and energy of the system.
"""

def symmetrize(a): # Symmetrize a matrix given a triangular one
    return a + a.T - np.diag(a.diagonal())

def eint(a,b,c,d): # Return compund index given four indices
    if a > b: ab = a*(a+1)/2 + b
    else: ab = b*(b+1)/2 + a
    if c > d: cd = c*(c+1)/2 + d
    else: cd = d*(d+1)/2 + c
    if ab > cd: abcd = ab*(ab+1)/2 + cd
    else: abcd = cd*(cd+1)/2 + ab
    return abcd

def tei(a,b,c,d): # two-election integral
    return twoe.get(eint(a,b,c,d),0.0)

# Put Fock matrix in Orthonormal AO basis
def fprime(X,F):
    return np.dot(np.transpose(X),np.dot(F,X))

def diagonalize(M): # Diagonalize a matrix. Return Eigenvalues
    e,Cprime = np.linalg.eigh(M)  # and non orthogonal Eigenvectors in separate 2D arrays.
    #e=np.diag(e)
    C = np.dot(S_minhalf,Cprime)
    return e,C # :D


def makedensity(C,P,dim,Nelec):
    """
    Make Density Matrix
    and store old one to test for convergence.
    """
    OLDP = np.zeros((dim,dim))
    for mu in range(0,dim):
        for nu in range(0,dim):
            OLDP[mu,nu] = P[mu,nu]
            P[mu,nu] = 0.0e0
            for m in range(0,Nelec//2):
                P[mu,nu] = P[mu,nu] + 2*C[mu,m]*C[nu,m]
    return P, OLDP
