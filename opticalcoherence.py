import numpy as np
from qutip import *
import pylab as plt

"""
This example demonstrates how to calculate a correlation function on the form ⟨𝐴(𝜏 )𝐵(0)⟩ for a non-steady initial
state. Consider an oscillator that is interacting with a thermal environment. If the oscillator initially is in a coherent
state, it will gradually decay to a thermal (incoherent) state. The amount of coherence can be quantified using the
first-order optical coherence function 𝑔
(1)(𝜏 ) = ⟨𝑎†(𝜏)𝑎(0)⟩√⟨𝑎†(𝜏)𝑎(𝜏)⟩⟨𝑎†(0)𝑎(0)⟩ .

For a coherent state |𝑔 (1)(𝜏 )| = 1, and for a completely incoherent (thermal) state 𝑔 (1)(𝜏 ) = 0. The following
code calculates and plots 𝑔(1)(𝜏 ) as a function of 𝜏 .

"""

N = 15
taus = np.linspace(0,10.0,200)
a = destroy(N)
H = 2 * np.pi * a.dag() * a

# collapse operator
G1 = 0.75
n_th = 2.00 # bath temperature in terms of excitation number
c_ops = [np.sqrt(G1 * (1 + n_th)) * a, np.sqrt(G1 * n_th) * a.dag()]

rho0 = coherent_dm(N, 2.0)
# first calculate the occupation number as a function of time
n = mesolve(H, rho0, taus, c_ops, [a.dag() * a]).expect[0]
# calculate the correlation function G1 and normalize with n to obtain g1
G1 = correlation_2op_2t(H, rho0, None, taus, c_ops, a.dag(), a)
g1 = G1 / np.sqrt(n[0] * n)
plot(taus, g1, 'b')
plot(taus, n, 'r')
title('Decay of a coherent state to an incoherent (thermal) state')
xlabel(r'$\tau$')
legend((r'First-order coherence function $g^{(1)}(\tau)$',
r'occupation number $n(\tau)$'))
show()

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
