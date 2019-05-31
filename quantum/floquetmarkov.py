import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import qutip as qt

"""
The QuTiP function qutip.floquet.fmmesolve implements the Floquet-Markov master equation. It calculates
the dynamics of a system given its initial state, a time-dependent hamiltonian, a list of operators through
which the system couples to its environment and a list of corresponding spectral-density functions that
describes the environment. In contrast to the qutip.mesolve and qutip.mcsolve, and the qutip.floquet.fmmesolve
does characterize the environment with dissipation rates, but extract the strength of the coupling to the
environment from the noise spectral-density functions and the instantaneous Hamiltonian parameters
(similar to the Bloch-Redfield master equation solver qutip.bloch_redfield.brmesolve).
"""

# Initalize parameters
delta = 0.0 * 2*np.pi
eps0 = 1.0 * 2*np.pi
A = 0.25 * 2*np.pi
omega = 1.0 * 2*np.pi
T = (2*np.pi)/omega
tlist = np.linspace(0.0, 20 * T, 101)

psi0 = basis(2,0)
H0 = - delta/2.0 * qt.sigmax() - eps0/2.0 * qt.sigmaz()
H1 = A/2.0 * qt.sigmax()
args = {"w": omega}
H = [H0, [H1, lambda t,args: np.sin(args["w"] * t)]
gamma1 = 0.1
def noise_spectrum(omega):
    """
    Noise power spectrum for angular frequency omega.
    """
    return 0.5 * gamma1 * omega/(2*np.pi)

# find the floquet modes for the time-dependent hamiltonian
f_modes_0, f_energies = qt.floquet_modes(H, T, args)

# precalculate mode table
f_modes_table_t = qt.floquet_modes_table(f_modes_0, f_energies, np.linspace(0, T, 500 + 1), H, T, args)

# solve the floquet-markov master equation
output = qt.fmmesolve(H, psi0, tlist, [sigmax()], [], [noise_spectrum], T, args)

# calculate expectation values in the computational basis
p_ex = np.zeros(np.shape(tlist), dtype=complex)
for idx, t in enumerate(tlist):
    f_modes_t = floquet_modes_t_lookup(f_modes_table_t, t, T)
    p_ex[idx] = expect(num(2), output.states[idx].transform(f_modes_t, True))

# For reference: calculate the same thing with mesolve
output = qt.mesolve(H, psi0, tlist, [np.sqrt(gamma1) * qt.sigmax()], [num(2)], args)
p_ex_ref = output.expect[0]

# plot the results
plt(tlist, real(p_ex), "r--", tlist, 1-real(p_ex), "b--")
plt(tlist, real(p_ex_ref), "r", tlist, 1-real(p_ex_ref), "b")
plt.xlabel("Time")
plt.ylabel("Occupation probability")
plt.legend(("Floquet $P_1$", "Floquet $P_0$", "Lindblad $P_1$", "Lindblad $P_0$"))
plt.show()
