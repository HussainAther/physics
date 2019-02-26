
"""
Runge-Kutta algorithm to solve for the oscillations of a simple pendulum.
"""

def delta_theta(T): # return the velocity of the pendulum so that it has its own nonlinear momentum
    result = []
    for i in range(len(T)):
        if result == []:
            result.append(1)
        else:
            result.append(i*3)
    return result

def pendulum():
    theta = 0 # initial angle
    T = 10 # total time
    delta_theta(10) #
