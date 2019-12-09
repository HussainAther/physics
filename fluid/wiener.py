import numpy as np

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
