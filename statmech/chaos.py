import numpy as np

"""
We use chaos to describe the qualitative change in the behavior of a solution as the solution patern
is not repetetive in any regular way. This aperiodic behavior of a solution
for a deterministic system that depends intimately on the intiial conditions
such that very small changes in the initial conditions can give rise to major 
differences in the solution at later times.

We can use a discrete nonlinear logistic-type model in which we assume
0 < u_0 < 1 for

u_t+1 = r * u_t * (1 - u_t) in which r > 0

for rate r, population function u over time t. 

The steady state solution and corresponding eiganvalues lambda are

u_star = 0 # we define u_star as this 0 equilibrium of the function of population 
lambda = f'(0) = r # r is the rate at t = 0. It's one of our solutions for lambda
u_star = (r-1)/r  # analytic solution
lambda = f'(u_star) = 2 - r

as r increases from 0 under the contsraint 0 < r < 1 
with non-negative equilibrium u_star 
"""

def ut2(u):
    """
    For some function u that depends upon t, return the u_t+2 value
    from the second iteration as r passes through the bifurcation value.
    """
