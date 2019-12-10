import numpy as np
import scipy.stats as stats

class ouprocess(p):
    """
    Orstein-Uhlenbeck process is a stochastic process that
    uses a mean reverting model. It's also called the Vasicek model.
    It uses the stochastic differential equation:
    
    dOU_t = theta * (m - OU_t) * dt + sigma * dB_t 
    """
    def __init__(self, 
                 theta, 
                 mu, 
                 sigma, 
                 startTime = 0, 
                 startPosition = 0, 
                 endPosition = None, 
                 endTime = None):
        assert sigma > 0 and theta > 0, "theta > 0 and sigma > 0."
        super(OU_process, self).__init__(startTime, startPosition, endTime, endPosition)
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.Normal = stats.norm()
