from sympy import symbols, latex, WildFunction, collect, Rational, simplify
from sympy.physics.secondquant import F, Fd, wicks, AntiSymmetricTensor, substitute_dummies, NO, evaluate deltas

"""
Define Hamiltonian and the second-quantized representation of a three-body Slater determinant.
"""

# Define Hamiltonian
p, q, r, s = symbols("p q r s", dummy=True)
f = AntiSymmetricTensor("f", (p,), (q,))
pr = Fd(p) * F(q)
v = AntiSymmetricTensor("v", (p, q), (r, s))
pqsr = Fd(p) * Fd(q) * F(s) * F(r)
Hamiltonian = f * pr + Rational(1) / Rational(4) * v * pqsr
a, b, c, d, e, f = symbols("a, b, c, d, e, f", above_fermi=True)

# Create teh representatoin
expression = wicks(F(c) * F(b) * F(a) * Hamiltonian * Fd(d) * Fd(e) * Fd(f), keep_only_fully_contracted=True, simplify_kronecker_deltas=True)
expression = evaluate_deltas(expression)
expression = simplify(expression)
print(latex(expression))
