import numpy as np
import scipy.integrate as integrate

"""
When an electromnagetic wave hits a charged particle, the electric and magneitc components of the wave
exert a Lorentz force on the particle and may set it into motion. The particle absorbs the energy
and re-emits electromagnetic radiation in a process of scattering known as Thomson scattering.

If we integrate the emission coefficient (ε) that describes the radially polarized light.
(ε dt dV dΩ dλ) is the energy scattered by a volume dV into time dt into a solid angle dΩ for a wavelength increase dλ.

For unpolarized light, ε can be divided into ε_t and ε_r .

ε_t = ((π σ_t)/2) In

ε_r = ((π σ_t)/2) In * cos(χ)^2

in which n is the density n of the charged particles at the scattering point, I is the incident flux,
and σ_t is the Thomson cross section for the charged particle. χ is the angle between incident and observed waves.

We can integrate epsilon across the solid angle dΩ
"""

esp_t = 5 # ε_t
eps_r = 6 # ε_r

result = integrate.quad(lambda x: (eps_t + eps_r)*np.sin(x), 0, np.pi) * integrate.quad(lambda x: 1, 0, 2*np.pi)

"""
result should be equal to I σ_t n (8/3) π ^2

We can get the Thomson differential cross section by

dσ_t/dΩ = (q^2 / (4 π ε_0 m c^2))^2 (1 + cos(χ)^2)/2
"""

e_o = 8.854e-12
c = 3e8

dsigdomeg = ((q**2)/(4*np.pi*e_o*m*c**2))**2 * ((1+np.cos(x)**2)/2)
