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
    return xdot[1][0]     
 
