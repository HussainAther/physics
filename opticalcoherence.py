import numpy as np
from qutip import *
import pylab as plt

"""
The second-order optical coherence function, with time-delay 𝜏 , is defined as

g(2)(𝜏) = <a†(0)a†(𝜏)a(𝜏)a(o)> / <a†(0)a(0)>^2

For a coherent state 𝑔(2)(𝜏) = 1, for a thermal state 𝑔 (2)(𝜏 = 0) = 2 and it decreases as a function of time
(bunched photons, they tend to appear together), and for a Fock state with 𝑛 photons 𝑔 (2)(𝜏 = 0) = 𝑛(𝑛−1)/𝑛2 <
1 and it increases with time (anti-bunched photons, more likely to arrive separated in time).

The following code calculates and plots 𝑔(2)(𝜏) as a function of 𝜏 for a coherent, thermal and fock state
"""


N = 25
taus = np.linspace(0, 25.0, 200)
a = destroy(N)
H = 2 * np.pi * a.dag() * a

kappa = 0.25
n_th = 2.0 # bath temperature in terms of excitation number
c_ops = [np.sqrt(kappa * (1 + n_th)) * a, np.sqrt(kappa * n_th) * a.dag()]
states = [{'state': coherent_dm(N, np.sqrt(2)), 'label': "coherent state"},
            {'state': thermal_dm(N, 2), 'label': "thermal state"},
            {'state': fock_dm(N, 2), 'label': "Fock state"}]

fig, ax = plt.subplots(1, 1)

for state in states:
    rho0 = state['state']
    # first calculate the occupation number as a function of time
    n = mesolve(H, rho0, taus, c_ops, [a.dag() * a]).expect[0]

    # calculate the correlation function G2 and normalize with n(0)n(t) to
    # obtain g2
    G2 = correlation_4op_1t(H, rho0, taus, c_ops, a.dag(), a.dag(), a, a)
    g2 = G2 / (n[0] * n)
    
    ax.plot(taus, np.real(g2), label=state['label'], lw=2)

ax.legend(loc=0)
ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$g^{(2)}(\tau)$')
plt.show()
