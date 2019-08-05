import numpy as np
import scipy.linalg

from foresttools import init_qvm_and_quilc
from grove.alpha.phaseestimation.phase_estimation import controlled
from grove.alpha.arbitrary_state.arbitrary_state import create_arbitrary_state
from pyquil.gates import CSWAP, H, MEASURE, SWAP
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

def rY(angle):
    """
    Generate a rotation matrix over the Y axis in the Bloch sphere.
    angle is the angle of rotation. 
    """
    return np.array([[np.cos(angle/2), -np.sin(angle/2)],
                     [np.sin(angle/2), np.cos(angle/2)]])

hhl.defgate("CRy0", controlled(rY(2pi/2**4)))
hhl += ("CRy0", 1, 0)
hhl.defgate("CRy1", controlled(rY(pi/2**4)))
hhl += ("CRy1", 2, 0)
hhl += uncomputation

reference_amplitudes = [0.9492929682, 0.3143928443]
# Target state preparation
hhl += create_arbitrary_state(reference_amplitudes, [4])
# Swap test
hhl += H(5)
hhl += CSWAP(5, 4, 3)
hhl += H(5)
c = hhl.declare("ro", "BIT", 2)
hhl += MEASURE(0, c[0])
hhl += MEASURE(5, c[1])

def get_psuccess(counts):
    """
    Compute the success probability of the HHL protocol from the statistics.
    """
    try:
        succ_rotation_fail_swap = counts["11"]
    except KeyError:
        succ_rotation_fail_swap = 0
    try:
        succ_rotation_succ_swap = counts["01"]
    except KeyError:
        succ_rotation_succ_swap = 0
    succ_rotation = succ_rotation_succ_swap + succ_rotation_fail_swap
    try:
        prob_swap_test_success = succ_rotation_succ_swap / succ_rotation
    except ZeroDivisionError:
        prob_swap_test_success = 0
    return prob_swap_test_success
