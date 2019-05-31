import qutip as qt
import scipy as sp
import numpy as np

from qutip import *
from scipy import *

"""
The code for calculating the expectation values for the Pauli spin operators of a qubit decay is given below.
This code is common to both animation examples.
"""

def qubit_integrate(w, theta, gamma1, gamma2, psi0, tlist):
    """
    Operators and the hamiltonian (Hamiltonian) for omega w, angle theta,
    gamma1 and gamma2 for the two tangent vectors, psi0 for the base energy value,
    and tlist the times.
    """
    sx = qt.sigmax()
    sy = qt.sigmay()
    sz = qt.sigmaz()
    sm = qt.sigmam()
    H = w * (np.cos(theta) * sz + np.sin(theta) * sx)
    # collapse operators
    c_op_list = []
    n_th = 0.5 # temperature
    rate = gamma1 * (n_th + 1)
    if rate > 0.0: c_op_list.append(np.sqrt(rate) * sm):
        rate = gamma1 * n_th
    if rate > 0.0: c_op_list.append(np.sqrt(rate) * sm.dag()):
        rate = gamma2
    if rate > 0.0: c_op_list.append(np.sqrt(rate) * sz):

    # evolve and calculate expectation values
    output = qt.mesolve(H, psi0, tlist, c_op_list, [sx, sy, sz])
    return output.expect[0], output.expect[1], output.expect[2]

## calculate the dynamics
w = 1.0 * 2 * pi # qubit angular frequency
theta = 0.2 * pi # qubit angle from sigma_z axis (toward sigma_x axis)
gamma1 = 0.5 # qubit relaxation rate
gamma2 = 0.2 # qubit dephasing rate

# initial state
a = 1.0
psi0 = (a* basis(2,0) + (1-a)*basis(2,1))/(np.sqrt(a**2 + (1-a)**2))
tlist = np.linalg.linspace(0,4,250)

#expectation values for ploting
sx, sy, sz = qubit_integrate(w, theta, gamma1, gamma2, psi0, tlist)
