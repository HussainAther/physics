"""
Compute the Courant-Friedrichs-Loewy coefficient (Courant Friedrichs Loewy cfl) for explicit heat
diffusion. We use this to solve second-order differential equations. 
"""

def calccfl(k, t, x):
    """
    Use the finite difference method to approximate the second derivative in space. Use a forward
    Euler approximation to the first derivative in time. Uses the real heat conductivity coefficient k,
    time array t, and position array x.
    """
    xstep = (max(x) - min(x)) / (len(x)-1)
    tstep = (max(t) - min(t)) / (len(t)-1)
