from sympy import symbols, latex, WildFunction, collect, Rational, simplify
from sympy.physics.secondquant import F, Fd, wicks, AntiSymmetricTensor, substitute_dummies, NO, evaluate deltas

"""
Define Hamiltonian and the second-quantized representation of a three-body Slater determinant.
"""

# Define Hamiltonian
p, q, r, s = symbols("p q r s", dummy=True)
