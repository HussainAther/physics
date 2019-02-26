import numpy as np
import matplotlib.pyplot as plt
from math import *

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
    y = np.zeros(n) # y coordinates
    v = np.zeros(n) # velocity values
    theta0 = 90 # initial angle
    # delta_theta(T) # velocity across the time interval. Still working on this
    for i in range(0, n-1):
        """
        Calculate Runge-Kutta formulations for each time point (step)
        """
        k1y = h*v[i]
        k1v = h*a(y[i])

        k2y = h*(v[i] + .5*k1v)
        k2v = h*a(y[i] + .5*k1y)

        k3y = h*(v[i] + .5*k2v)
        k3v = h*a(y[i] + .5*k2y)

        k4y = h*(v[i] + k3v)
        k4v = h*a(y[i] + k4y)

        y[i+1] = y[i] + (k1y + 2 * k2y + 2 * k3y + k4y) / 6.0
        v[i+1] = v[i] + (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0

t = np.range(0, c, dt)
y[0] = np.radians(theta0)
v[0] = np.radians(0)
pendulum()

plt.plot(t, y)
plt.title('Pendulum Motion with Runge-Kutta method:')
plt.xlabel('time (s)')
plt.ylabel('angle (rad)')
plt.grid(True)
plt.show()

"""
Now solve the pendulum using Euler method.
"""
def euler(dt, yi, theta):
    """
    Euler step uses -sin(theta) to solve differential equations
    """
    dtheta = dt * yi
    dy = dt * (-sin(theta))
    yi += dy
    theta += dtheta
    return yi

# initial positions
theta = .5
y = [] # list of y coordinates
y0 = 0 # initial y coordinate
c = 10 # number of cycles
dt = .05 # time step (interval)
t = np.range(0, c, dt)


for i in t:
    (y0, theta) = euler(dt, y0, theta)
    y.append(y0)

plt.plot(t, y)
plt.title('Pendulum Motion with Euler method:')
plt.xlabel('time (s)')
plt.ylabel('angle (rad)')
plt.grid(True)
plt.show()
