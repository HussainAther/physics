import numpy as np
from gekko import GEKKO
import matplotlib.pyplot as plt

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
N_A = m.var(50) # number of particles of gas A
N_B = m.var(60) # of gas B
F_A = m.var(10) # energy of gas A
F_B = m.var(20) # energy of gas B
m.Equation(F_A.dN_A * dN_A + F_B.dN_B * dN_B = dF)
m.options.IMODE=4
m.solve()


"""
From this we define chemical potential as

μ(T, V, N) = (∂F/∂N)_V,T

the potential equals the change in energy as number of moles of gas changes with volume and temperature constant

"""
