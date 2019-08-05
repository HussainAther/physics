import numpy as np
from functools import partial
from pyquil import Program, api
from pyquil.paulis import PauliSum, PauliTerm, exponential_map, sZ
from pyquil.gates import
from scipy.optimize import minimize
from foresttools import

"""
Quantum approximate optimization algorithm for a circuit.
"""

np.set_printoptions(precision=3, suppress=True)
qvm_server, quilc_server, fc = init_qvm_and_quilc("")
n_qubits = 2
# Hamiltonian
Hm = [PauliTerm("X", i, -1.0) for i in range(n_qubits)]
J = np.array([[0,1],[0,0]]) # weight matrix of the Ising model. Only the coefficient (0,1) is non-zero.

Hc = []
for i in range(n_qubits):
    for j in range(n_qubits):
        Hc.append(PauliTerm("Z", i, -J[i, j]) * PauliTerm("Z", j, 1.0))

# Iterate to compute exponential functions.
exp_Hm = []
exp_Hc = []
for term in Hm:
    exp_Hm.append(exponential_map(term))
for term in Hc:
    exp_Hc.append(exponential_map(term))
