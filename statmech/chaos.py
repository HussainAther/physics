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

for bifurcation value r, population function f that depends upon some function
u that depends upon t.  

The steady state solution and corresponding eiganvalues lambda are

u_star = 0 # we define u_star as this 0 equilibrium of the function of population 
lambda = f'(0) = r # r is the rate of change of the function f at u = 0. It's one of our solutions for lambda
u_star = (r-1)/r  # analytic solution
lambda = f'(u_star) = 2 - r

as r increases from 0 under the contsraint 0 < r < 1 
with non-negative equilibrium u_star 
"""
# central differentiation
def cd(y, t, h):
    """
    Differentiate around both sides of a point t of interval h.
    """
    return ( y(t+h/2) -y(t-h/2))/h

"""
We can observe what happens to our functions as we vary our bifurcation value (r) and 
let the solution pass over various values including r. We use an iterative procedure
in which we define the various subtitles of u to be the number of times we 
differentiate f around t = u0 in which u0 is our initial condition. 
"""

def ut2(f, t):
    """
    For some function f that depends upon u that depends upon t, return the u_t+2 value
    from the second iteration as r passes through the bifurcation value.
    """
    r = cd(u(f, 0, 5)) # over some interval t of size 5, differentiation the function f around u = 0
    return r * (u(t) * (t-u(t)) * (1 - r*u(t) * (1 - u(t))))
