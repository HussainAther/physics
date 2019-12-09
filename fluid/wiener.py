

class wienerprocess():
    """
    Wiener process implementation as a special case of brownian motion
    with 0 drift and variance of t.
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
