from sympy import sqrt, I, Symbol, Integer, S
from sympy.functions.special.tensor_functions import KroneckerDelta
import sympy.physics.quantum as sm

"""
Simple Harmonic Oscillator in 1-Dimension
"""
class SHOOp(Operator):
    """
    SHO Operators
    """
    @classmethod
    def _eval_args(cls, args):
        args = QExpr._Eval_args(args)
        if len(args) == 1:
            return args
        else:
            raise ValueError("Only one argument")

    @classmethod
    def _eval_hilbert_space(cls, label):
        return sm.hilbert.ComplexSpace(S.Infinity)
