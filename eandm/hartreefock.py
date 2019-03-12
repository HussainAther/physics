from __future__ import division
from numpy import genfromtxt

import sys
import math
import numpy as np
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
HF wave function and energy of the system.
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


def makedensity(c, p, dim, Nelec):
    """
    Make Density Matrix
    and store old one to test for convergence.
    """
    oldp = np.zeros((dim,dim))
    for mu in range(0,dim):
        for nu in range(0,dim):
            oldp[mu,nu] = p[mu,nu]
            p[mu,nu] = 0.0e0
            for m in range(0,Nelec//2):
                p[mu,nu] += 2*c[mu,m]*c[nu,m]
    return p, oldp

def makefock(Hcore,p,dim): # Make Fock Matrix
    f = np.zeros((dim,dim))
    for i in range(0,dim):
        for j in range(0,dim):
            f[i,j] = Hcore[i,j]
                for k in range(0,dim):
                    for l in range(0,dim):
                        f[i,j] += p[k,l]*(tei(i+1,j+1,k+1,l+1)-0.5e0*tei(i+1,k+1,j+1,l+1))
    return f

def deltap(o, oldp):
    """
    Calculate change in density matrix
    """
    delta = 0.0e0
    for i in range(0,dim):
        for j in range(0,dim):
            delta += ((p[i,j]-oldp[i,j])**2)
    delta = (delta/4)**(0.5)
    return delta

def currentenergy(p, Hcore, f ,dim):
    """
    Calculate energy at iteration
    """
    en = 0
    for mu in range(0,dim):
        for nu in range(0,dim):
            en += 0.5*p[mu,nu]*(Hcore[mu,nu] + f[mu,nu])
    return en

# Define variables and run the functions.
# ENUC = nuclear repulsion, Sraw is overlap matrix, Traw is kinetic energy matrix,
# Vraw is potential energy matrix

Nelec = 2

# Read in raw data from files in directory
ENUC = genfromtxt('./enuc.dat',dtype=float,delimiter=',')
Sraw = genfromtxt('./s.dat',dtype=None)
Traw = genfromtxt('./t.dat',dtype=None)
Vraw = genfromtxt('./v.dat',dtype=None)

# dim is the number of basis functions
dim = int((np.sqrt(8*len(Sraw)+1)-1)/2)

# Initialize integrals, and put them in convenient Numpy array format
S = np.zeros((dim,dim))
T = np.zeros((dim,dim))
V = np.zeros((dim,dim))

for i in Sraw: S[i[0]-1,i[1]-1] = i[2]
for i in Traw: T[i[0]-1,i[1]-1] = i[2]
for i in Vraw: V[i[0]-1,i[1]-1] = i[2]

# The matrices are stored triangularly. For convenience, we fill
# the whole matrix. The function is defined above.
S = symmetrize(S)
V = symmetrize(V)
T = symmetrize(T)

Hcore = T + V

"""
Like the core hamiltonian, we need to grab the integrals from the
separate file, and put into ERIraw (ERI = electron repulsion integrals).
I chose to store the two electron integrals in a python dictionary.
The function 'eint' generates a unique compund index for the unique two
electron integral, and maps this index to the corresponding integral value.
'twoe' is the name of the dictionary containing these.
"""

ERIraw = genfromtxt('./eri.dat',dtype=None)
twoe = {eint(row[0],row[1],row[2],row[3]) : row[4] for row in ERIraw}

# Orthogonalize the basis using symmetric orthogonalization and S^(-1/2) as the
# transformation matrix.
SVAL, SVEC = np.linalg.eig(S)
SVAL_minhalf = (np.diag(SVAL**(-0.5)))
S_minhalf = np.dot(SVEC,np.dot(SVAL_minhalf,np.transpose(SVEC)))

P = np.zeros((dim,dim)) # P is density matrix, set intially to zero.
delta = 1.0
convergence = 0.00000001
G = np.zeros((dim,dim)) # The G matrix is used to make the Fock matrix
while delta > convergence:
    F = makefock(Hcore,P,dim)
   # print "F = \n", F
    Fprime = fprime(S_minhalf,F)
   # print "Fprime = \n", Fprime

    E,Cprime = np.linalg.eigh(Fprime)
    C = np.dot(S_minhalf,Cprime)
    E,C = diagonalize(Fprime)
   # print "C = \n", C

    P,OLDP = makedensity(C,P,dim,Nelec)

   # print "P = \n", P

    delta = deltap(P,OLDP)
   #print "E= ",currentenergy(P,Hcore,F,dim)+ENUC
   #print "delta = ", delta,"\n"

    EN = currentenergy(P,Hcore,F,dim)
    print "TOTAL E(SCF) = \n", EN + ENUC
   #print "C = \n", C
