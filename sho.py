"""Simple Harmonic Oscillator 1-Dimension"""

from sympy import sqrt, I, Symbol, Integer, S
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum.operator import Operator
from sympy.physics.quantum.state import Bra, Ket, State
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.cartesian import X, Px
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.physics.quantum.matrixutils import matrix_zeros
from sympy.physics.quantum.represent import represent
