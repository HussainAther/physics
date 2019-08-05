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

n_iter = 10 # number of iterations of the optimization procedure
p = 1
beta = np.random.uniform(0, np.pi*2, p)
gamma = np.random.uniform(0, np.pi*2, p)
initial_state = Program()
for i in range(n_qubits):
    initial_state += H(i)

def create_circuit(beta, gamma):
    circuit = Program()
    circuit += initial_state
    for i in range(p):
        for term_exp_Hc in exp_Hc:
            circuit += term_exp_Hc(-beta[i])
        for term_exp_Hm in exp_Hm:
            circuit += term_exp_Hm(-gamma[i])
    return circuit

def evaluate_circuit(beta_gamma):
    beta = beta_gamma[:p]
    gamma = beta_gamma[p:]
    circuit = create_circuit(beta, gamma)
    return qvm.pauli_expectation(circuit, sum(Hc))

qvm = api.QVMConnection(endpoint=fc.sync_endpoint, compiler_endpoint=fc.compiler_endpoint)

result = minimize(evaluate_circuit, np.concatenate([β, γ]), method="L-BFGS-B")

circuit = create_circuit(result["x"][:p], result["x"][p:])
