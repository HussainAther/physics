import numpy as np
import scipy.stats as stats

from wiener import wienerprocess 

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

    def _mean(self, t):
        """
        Return the mean at a time position t.
        """
        if not self.conditional:
            return self.startPosition*np.exp(self.mu*t)
        else:
           delta = self.endPosition - self.startPosition
           return self.startPosition*np.exp(-0.5*(self.sigma*t)**2/delta+self.Weiner.endPosition*t/delta)        
            
    def _transition_pdf(self,x,t,y):
        """
        Transition function for given probabilities.
        """
        delta = t - self.startTimee
        d = (self.mu - 0.5*self.sigma**2)
        return  x/(y*np.sqrt(2*Pi*self.sigma**2**delta))*np.exp(-(np.log(y/x)-d)**2/(2*self.sigma**2*delta))
    
    def _var(self,t):
        """
        Variance.
        """
        if not self.conditional:
            return self.startPosition**2*np.exp(2*self.mu*t)*(np.exp(self.sigma**2*t)-1)
        else:
            X=np.log(self.endPosition/self.startPosition)-(self.mu-0.5*(self.sigma**2))*T
            X=X/self.sigma
            delta = self.endTime-self.startTime
            return self.startPosition**2*np.exp(2*self.mu*t-self.sigma**2*t + self.sigma*t/delta*(self.sigma*(delta-T)+2*X))*(np.exp(self.sigma**2*(delta-t)*t/delta)-1)

    def drift(self, x, t):
        return self.mu*x
        
    def diffusion(self, x, t):
        return self.sigma*x
