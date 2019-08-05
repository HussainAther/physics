import numpy as np
import scipy.linalg

from foresttools import init_qvm_and_quilc
from grove.alpha.phaseestimation.phase_estimation import controlled
from pyquil.gates import H, SWAP
from pyquil import Program, get_qc

"""
Harrow-Hassidim-Lloyd (HHL) algorithm.
"""

qvm_server, quilc_server, fc = init_qvm_and_quilc("")
qc = get_qc("6q-qvm", connection=fc)
pi = np.pi
A = 0.5*np.array([[3, 1], [1, 3]])
hhl = Program()

# Quantum phase estimation through superposition
hhl += H(1)
hhl += H(2)

# Controlled-U0
hhl.defgate("CONTROLLED-U0", controlled(scipy.linalg.expm(2j*π*A/4)))
hhl += ("CONTROLLED-U0", 2, 3)

# Controlled-U1
hhl.defgate("CONTROLLED-U1", controlled(scipy.linalg.expm(2j*π*A/2)))
hhl += ("CONTROLLED-U1", 1, 3)

# Inverse quantum inverse Fourier transformation
hhl += SWAP(1, 2)
hhl += H(2)
hhl.defgate("CSdag", controlled(np.array([[1, 0], [0, -1j]])))
hhl += ("CSdag", 1, 2)
hhl += H(1)
hhl += SWAP(1, 2)
uncomputation = hhl.dagger()
