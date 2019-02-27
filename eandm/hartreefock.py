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
