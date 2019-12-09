import numpy as np
import scipy.stats as stats

class wienerprocess():
    """
    Wiener process implementation as a special case of Brownian motion
    with 0 drift and variance of t. Homogenous Markov process with a transition
    function.
    dW_t = mu*dt + sigma*dB_t
    W_t ~ N*(mu*t, sigma**2t)
    """
    def __init__(self,  
                 mu, 
                 sigma, 
                 startTime = 0, 
                 startPosition = 0, 
                 endPosition = None, 
                 endTime = None):
        super(wienerprocess,self).__init__(startTime, startPosition, endTime, endPosition)
        self.mu = mu
        self.sigma = sigma
        self.Nor = stats.norm()

    def _transition_pdf(self, 
                        x, 
                        t, 
                        y):
        """
        Transition occurs between states.
        """
        return np.exp(-(y-x-self.mu*(t-self.startTime))**2/(2*self.sigma**2*(t-self.startTime)))\
            /np.sqrt(2*pi*self.sigma*(t-self.startTime))   

    def _mean(self,
              t):
        """
        Compute the mean position given the start and end times.        
        """
        if self.conditional:
            delta1 = t - self.startTime
            delta2 = self.endTime - self.startTime
            return self.startPosition + self.mu*delta1 + (self.endPosition-self.startPosition-self.mu*delta2)*delta1/delta2
        else:
            return self.startPosition+self.mu*(t-self.startTime)

    def _var(self,
             t):
        """
        Compute variance.
        """
        if self.conditional:
            delta1 = self.sigma**2*(t-self.startTime)*(self.endTime-t)
            delta2 = self.endTime-self.startTime
            return delta1/delta2
        else:
            return self.sigma**2*(t-self.startTime)

    def _sample_position(self,t, n=1):
        """
        This incorporates both conditional and unconditional
        """
        return self.mean(t) + np.sqrt(self.var(t))*self.Nor.rvs(n) 
