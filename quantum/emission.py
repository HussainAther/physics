import numpy as np
import pylab as plt

from qutip import *

"""
Given a correlation function ⟨𝐴(𝜏 )𝐵(0)⟩ we can define the corresponding power spectrum as

𝑆(𝜔) = ∫︁<𝐴(𝜏)𝐵(0)> exp(−𝑖𝜔𝜏)𝑑𝜏.

In QuTiP, we can calculate 𝑆(𝜔) using either qutip.correlation.spectrum_ss, which first calculates
the correlation function using the qutip.essolve.essolve solver and then performs the Fourier transform
semi-analytically, or we can use the function qutip.correlation.spectrum_correlation_fft to
numerically calculate the Fourier transform of a given correlation data using FFT.
The following example demonstrates how these two functions can be used to obtain the emission power spectrum.
"""

N = 4 # number of cavity fock states
wc = wa = 1.0 * 2 * np.pi # cavity and atom frequency
g = 0.1 * 2 * np.pi # coupling strength
kappa = 0.75 # cavity dissipation rate
gamma = 0.25 # atom dissipation rate
# Jaynes-Cummings Hamiltonian
a = tensor(destroy(N), qeye(2))
sm = tensor(qeye(N), destroy(2))
H = wc * a.dag() * a + wa * sm.dag() * sm + g * (a.dag() * sm + a * sm.dag())
# collapse operators
n_th = 0.25
c_ops = [np.sqrt(kappa * (1 + n_th)) * a, np.sqrt(kappa * n_th) * a.dag(), np.sqrt(gamma) * sm]

"""
calculate the correlation function using the mesolve solver, and then fft to
obtain the spectrum. Here we need to make sure to evaluate the correlation
function for a sufficient long time and sufficiently high sampling rate so
that the discrete Fourier transform (FFT) captures all the features in the
"""

# resulting spectrum.
tlist = np.linspace(0, 100, 5000)
corr = correlation_ss(H, tlist, c_ops, a.dag(), a)
wlist1, spec1 = spectrum_correlation_fft(tlist, corr)

# calculate the power spectrum using spectrum, which internally uses essolve
# to solve for the dynamics (by default)
wlist2 = np.linspace(0.25, 1.75, 200) * 2 * np.pi
spec2 = spectrum(H, wlist2, c_ops, a.dag(), a)

# plot the spectra
fig, ax = plt.subplots(1, 1)
ax.plot(wlist1 / (2 * np.pi), spec1, "b", lw=2, label='eseries method')
ax.plot(wlist2 / (2 * np.pi), spec2, "r--", lw=2, label='me+fft method')
ax.legend()
ax.set_xlabel("Frequency")
ax.set_ylabel("Power spectrum")
ax.set_title("Vacuum Rabi splitting")
ax.set_xlim(wlist2[0]/(2*np.pi), wlist2[-1]/(2*np.pi))
plt.show()
