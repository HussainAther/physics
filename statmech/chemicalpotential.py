import numpy as np
import matplotlib.pyplot as plt

from gekko import GEKKO

"""
To address thermodynamic systems open to diffusion, we need to identify an equilibrium
such that there is no net flow of particles between two systems.
"""

def helmholtz(a, b):
    """
    Return the Helmholtz free energy for a combined system of two different gas systems a and b
    such that particles can bass from a to b and the other way. This is the sum of the energy
    for the components Fa + Fb.
    """
    return a + b

m = GEKKO() # initialize GEKKO model
N_A = m.Var(50) # number of particles of gas A
N_B = m.Var(60) # of gas B
F_A = m.Var(10) # energy of gas A
F_B = m.Var(20) # energy of gas B
m.Equation(F_A.N_A * N_A + F_B.N_B * N_B == dF)
m.options.IMODE=4
m.solve()


"""
From this we define chemical potential as (mu)

μ(T, V, N) = (∂F/∂N)_V,T

the potential equals the change in energy as number of moles of gas changes with volume and temperature constant

Gibbs free energy is G(T,p,N) = E - TS + pV with its differential

dG = -Sdt + Vdp + μdN

For different gas particles interacting with one another, the total gibbs free energy can be calculated.
"""

def N(i, mu):
    """
    N is the number of particles of teh gas. Let this be an expansion of a gas with respect to the time Variable i.
    """
    return 3 * i**2

mu = {"x": 5, # chemical potential dictionary for hypothetical gases x, y, and c
    "y": 19,
    "y": 24
}


def gibbsMultiple(particle_list, k):
    """
    Calculate Gibbs free energy across several different species.
    """
    result = 0
    for j in particle_list:
        result += N(i, mu[j]) * mu[j]
    return result

