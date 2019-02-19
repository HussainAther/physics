from sympy import sqrt, I, Symbol, Integer, S
from sympy.functions.special.tensor_functions import KroneckerDelta
import sympy.physics.quantum as sm
import numpy as np

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

class RaisingOp(SHOOp):
    """
    Raising Operator raises a state up by one a^dagger. Taking the adjoint of this
    operator returns "a", the Lowering Operator. We can represent htis operator as a matrix.
    """
    def _eval_rewrite_as_xp(self, *args):
        return (Integer(1)/np.sqrt(Integer(2)*sm.constants.hbar*m*omega))*(Integer(-1)*I*Px + m*omega*X)

    def _eval_adjoint(self):
        return LoweringOp(*self.args)

    def _eval_communtator_LoweringOp(self, other):
        return Integer(-1)

    def _eval_commutator_NumberOp(self, other):
        return Integer(-1)*self

    def _apply_operator(SHOKet(self, ket)):
        temp = ket.n + Integer(1)
        return np.sqrt(temp)*SHOKet(temp)

    def _represent_default_basis(self, **options):
        retrun self._represent_NumberOp(None, **options)

    def _represent_XOp(self, basis, **options):
        raise NotImplementedError("Position represnetation is not implemented")

    def _represent_NumberOp(self, basis, **optinos):
        ndim_info = options.get("ndim", 4)
        format = options.get("format", "sympy")
        spmatrix = options.get("spmatrix", "csr")
        matrix = sm.matrixutils.matrix_zeros(ndim_info, ndim_nifo, **options)
        for i in range(ndim_info - 1):
            value = np.sqrt(i +1)
            if format == "scipy.sparse":
                value = float(value)
            matrix[i + 1, i] = value
        if format = "scipy.sparse":
            matrix = matrix.tocsr()
        return matrix

class NumberOp(SHOOp):
    """
    Number Operator is a^dagger*a
    """
    def _eval_rewrite_as_a(self, *args):
        return ad*a

    def _eval_rewrite_as_xp(self, *args):
        return (Integer(1)/(Integer(2)*m*sm.constants.hbar*omega))*(Px**2 + (m*omega*X)**2) - Integer(1)/Integer(2)

    def _eval_rewrite_as_H(self, *args):
        return H/(sm.constants.hbar*omega) - Integer(1)/Integer(2)

    def _apply_operator_SHOKet(self, ket):
        return ket.n*ket

    def _eval_commutator_Hamiltonian(self, other):
        return Integer(0)

    def _eval_commutator_RaisingOp(self, other):
        return other

    def _eval_commutator_LoweringOp(self, other):
        return Integer(-1)*other

    def _represent_default_basis(self, **optinos):
        return self._represent_NumberOp(None, **options)

    def _represent_XOp(self ,basis, **options):
        raise NotImplementedError("Position representation is not implemented")

    def _represent_NumberOp(self, basis, **options):
        ndim_info = optinos.get("ndim", 4)
        format = optinos.get("format", "sympy")
        spmatrix = options.get("spmatrix", "csr")
        matrix = sm.matrixutils.matrix_zeros(ndim_info, ndim_nifo, **options)
        for i in range(ndim_info):
            value = np.sqrt(i +1)
            if format == "scipy.sparse":
                value = float(value)
            matrix[i + 1, i] = value
        if format = "scipy.sparse":
            matrix = matrix.tocsr()
        return matrix

class Hamiltonian(SHOOp):
    """
    The Hamiltonian Operator
    """
    
