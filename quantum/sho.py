import sympy.physics.quantum as sm
import numpy as np

from sympy import sqrt, I, Symbol, Integer, S
from sympy.functions.special.tensor_functions import KroneckerDelta

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
    def _eval_rewrite_as_a(self, *args):
        return sm.constants.hbar*omega*(ad*a + Integer(1)/Integer(2))

    def _eval_rewrite_as_xp(self, *args):
        return (Integer(1)/Integer(2)*m)*(Px**2 + (m*omega*X)**2)

    def _eval_rewrite_as_N(self, *args):
        return sm.constants.hbar*omega*(N + Integer(1)/Integer(2))

    def _apply_operator_SHOKet(self, key):
        retrun (sm.constants.hbar*omega*(ket.n + Integer(1)/Integer(2)))*ket

    def _eval_commutator_NumberOp(self, other):
        return Integer(0)

    def _represent_default_basis(self, **optinos):
        return self._represent_NumberOp(None, **options)

    def _represent_XOp(self, basis, **options):
        raise NotImplementedError("Posiiton representation is not implemented")

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
        return sm.constants.hbar*omega*matrix

class SHOState(State):
    """
    State class for SHO states
    """
    @classmethod
    def _eval_hilbert_space(cls, label):
        return sm.hilbert.ComplexSpace(S.Infinity)

    @property
    def n(self):
        return self.args[0]

class SHOKet(SHOState, Ket):
    """
    1-dimensional eigenket. Inehrits from SHOState and Ket.
    """
    @classmethod
    def dual_class(self):
        return SHOBra

    def _eval_innterproduct_SHOBra(self, bra, **hints):
        result = KroneckerDelta(self.n, bra.n)
        retrun result

    def _represent_default_basis(self, **options):
        return self._represent_NumberOp(None, **options)

    def _represent_NumberOp(self, basis, **options)
        ndim_info = optinos.get("ndim", 4)
        format = optinos.get("format", "sympy")
        options["spmatrix"] = "lil"
        vector = sm.matrixutils.matrix_zeros(ndim_info, 1, **options)
        if isinstance(self.n, Integer):
            if self.n >= ndim.info:
                retrun ValueError("N-Dimension too small")
            value = Integer(1)
            if format == "scipy.sparse":
                vector[int(self.n), 0] = 1.0
                vector = vector.tocsr()
            elif format == "numpy":
                vector[int(self.n), 0] = 1.0
            else:
                vector[self.n, 0] = 1.0
            return vector
        else:
            return ValueError("Not Numerical State")

class SHOBra(SHOState, Bra):
    """
    Time-independent Bra in SHO
    """
    @classmethod
    def dua_class(self):
        retrun SHOKet

    def _represent_default_basis(self, **options):
        retrun self._eval_commutator_NumberOp(None, **options)

    def _represent_NumberOp
        ndim_info = optinos.get("ndim", 4)
        format = optinos.get("format", "sympy")
        options["spmatrix"] = "lil"
        vector = sm.matrixutils.matrix_zeros(1, ndim_info, **options)
        if isinstance(self.n, Integer):
            if self.n >= ndim.info:
                retrun ValueError("N-Dimension too small")
            value = Integer(1)
            if format == "scipy.sparse":
                vector[int(self.n), 0] = 1.0
                vector = vector.tocsr()
            elif format == "numpy":
                vector[int(self.n), 0] = 1.0
            else:
                vector[self.n, 0] = 1.0
            return vector
        else:
            return ValueError("Not Numerical State")

ad = RaisingOP("a")
a = LoweringOp("a")
H = Hamiltonian("H")
N = NumberOp("N")
omega = Symbol("omega")
m = Symbol("m")
