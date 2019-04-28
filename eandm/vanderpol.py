import numpy as np

"""
We can create electrical circuits governed by the van der Pol oscillator equation:

dx2/d2t - epsilon(1-x^2)dx/dt + x = 0

in which epsilon is a parameter > 0 for some dynamical variable x.
"""

def response(x, t):
    """
    The van der Pol oscillator responds to periodic forcing with two frequencies
    in the system: frequency of self-oscillation and frequency of periodic forcing
    for some input lists x and t of position and time, respectively.
    """
    xdot = ["", ""] # first and second derivative of x
    e = 10 # epsilon
    xdot[0] = x[0]
    xdot[1] = x[0] /(t[1]- t[0]) 
    return xdot[1] - e(1 - x[0]**2) * xdot[0] + x[0] 

def psi(x):
    """
    Output for tunnel diode.
    """
    return np.sin(x)

def circuit(V, I0, L, C, alpha, beta):
    """
    For a list of input voltage V, initial input current I0, inductance L and capacitance C, we can introduce new 
    variables alpha and gamma to study an electrical circuit with a tunnel diode for the van der Pol oscillator.
    """
    Vdot = [V[0], V[0]/(psi(np.pi(/2))) # first and second derivative of V
    return Vdot[1] - (1/C) * (alpha - 3*gamma*V[0]**2)*Vdot[0] + 1/(L*C)*V[0] 
     
