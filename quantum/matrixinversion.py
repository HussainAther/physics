import numpy as np
import scipy.linalg

from grove.alpha.phaseestimation.phase_estimation import controlled
from pyquil import Program, get_qc, init_qvm_and_quilc

"""
Harrow-Hassidim-Lloyd (HHL) algorithm.
"""

qvm_server, quilc_server, fc = init_qvm_and_quilc("")
qc = get_qc("6q-qvm", connection=fc)
pi = np.pi
