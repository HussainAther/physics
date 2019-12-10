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

    def _mean(self,
              t):
        """
        Average at time t
        """
        if self.conditional:
            return super(OU_process,self)._mean(t) 
        else:
            return self.startPosition*np.exp(-self.theta*(t-self.startTime))+self.mu*(1-np.exp(-self.theta*(t-self.startTime)))
                                                                
    def _var(self,
             t):
        """
        Variance
        """
        if self.conditional:
            return super(OU_process,self)._get_variance_at(t)
        else:
            return self.sigma**2*(1-np.exp(-2*self.theta*t))/(2*self.theta)
    
    def _transition_pdf(self,
                        x,
                        t,
                        y):
        """
        Transition from x to y
        '""
            mu = x*np.exp(-self.theta*t)+self.mu*(1-np.exp(-self.theta*t))
            sigmaSq = self.sigma**2*(1-np.exp(-self.theta*2*t))/(2*self.theta)
            return np.exp(-(y-mu)**2/(2*sigmaSq))/np.sqrt(2*pi*sigmaSq)            
