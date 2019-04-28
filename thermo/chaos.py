import numpy as np

"""
We observe chaos when we split separatrices of Hamiltonian systems in homo-
and heteroclinic orbits.
"""

def eq(x, t, w, e):
    """
    An equation of motion for which we will find the coresponding Hamiltonian
    vector field for positions x, times t, omega (angular frequency) w, and 
    epsilon (eigenvector) e.
    """
    y = x[0]
    dy = x[1]
    xdot = [[],[]] # first and second derivative of x
    xdot[0][0] = dy # first derivate of x
    """
    Introduce our equation of motion which we can solve by setting
    equal to zero.
    """
    xdot[1][0] = -w**2 + e * np.cos(t[1]) * np.sin(x[1])
    """
    Solving this using the Hamiltonian vector field, we obtain a system
    of equations. 
    """
    tdot = [1]*len(t) # differential of time 
    xdot[0][0] = y # set the first derivative equal to y, which should be x[0] in the original input
    ydot = -(w**2 + e * np.cos(t) * np.sin(x[0]) 

"""
We observe chaos in this Hamiltonian dynamical system.
""" 
