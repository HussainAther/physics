import numpy as np

"""
Brownian motion process by stochastic differential
equation (SDE)
 
dGM_t = mu*GMB_tdt + sigma*GM_t*dW_t

with solution 

GBM_t = GBM_0 * exp((mu -.5*sigma^2)*t + sigma*W_t)
"""
