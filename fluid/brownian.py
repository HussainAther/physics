import numpy as np
import scipy.stats as stats

class gbm():
    """
    Geometric Brownian motion process by stochastic 
    differential equation (SDE)
     
    dGM_t = mu*GMB_tdt + sigma*GM_t*dW_t
    
    with solution 
    
    GBM_t = GBM_0 * exp((mu -.5*sigma^2)*t + sigma*W_t)
    """
    def __init__(self, 
                 mu, 
                 sigma, 
                 startPosition = 1, 
                 startTime = 0, 
                 endPosition = None, 
                 endTime = None):
        super(GBM_process,self).__init__(startTime, startPosition, endTime, endPosition)
        self.mu = mu
        self.sigma = sigma
        self.Nor = stats.norm(0,1)

    def drift(self, x, t):
        return self.mu*x
        
    def diffusion(self, x, t):
        return self.sigma*x
