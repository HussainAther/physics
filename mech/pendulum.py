
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

def pendulum():
    theta = 0 # initial angle
    T = 10 # total time
    delta_theta(T) # velocity across the time interval
    for t in range(0, T):  # loop over time interval
        k1 = 
