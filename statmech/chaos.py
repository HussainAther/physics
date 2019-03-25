import numpy as np

"""
We can use a discrete nonlinear logistic-type model in which we assume
0 < u_0 < 1 for

u_t+1 = r * u_t * (1 - u_t) in which r > 0

for rate r, population function u over time t. 

The steady state solution and corresponding eiganvalues lambda are

u_star = y0 
lambda = f'(0) = r
u_star = (r-1)/r 
lambda = f'(ustar) = 2 - r

"""
