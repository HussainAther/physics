
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
    H = a
