import numpy as np
import matplotlib.pyplot as plt

"""
Runge-Kutta algorithm to solve for the oscillations of a simple pendulum.
"""

def delta_theta(T):
    """
    return the velocity of the pendulum so that it has its own nonlinear momentum
    """
    result = []
    for i in range(len(T)):
        if result == []:
            result.append(1)
        else:
            result.append(i*3)
    return result

def delta_delta_theta(theta, delta_theta):
    """
    return the angular acceleration. with the absence of friction
    and external torques, this takes the form:
    
    d^2(theta)/dt^2 = -(delta_theta*r)^2 * sin(theta)
    """
    return -(delta_theta)^2 * sin(theta)

def a(theta):
    """
    Acceleration due to gravity.
    """
    return -(m*g*r/I) * np.sin(theta)

def pendulum():
    m = 3.0 # mass
    g = 9.8 # acceleration due to gravity
    r = 2.0 # radius (length)
    I = 12.0 # moment of Inertia
    dt = .0025 # step size
    l = 2.0
    c = 10 # number of cycles
    t = np.range(0, c, dt) # range of iterations for each step size
    n = len(t) # number of iterations
    theta = 0 # initial angle
    # delta_theta(T) # velocity across the time interval. Still working on this
    v = dy/dt # calculate velocity
    acceleration = dv/dt
    for t in range(0, T):  # loop over time interval
        k1 =
