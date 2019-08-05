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
