import numpy as np
import scipy.stats as stats

class poissonprocess(p)
    """
    Poisson process with rate parameter
    """
    def __init__(self,
                 rate=1, 
                 startTime = 0, 
                 startPosition = 0, 
                 endPosition = None, 
                 endTime = None):
       assert rate > 0, "invalid parameter: rate parameter must be greater than 0."
       self.rate = rate
       self.Exp = stats.expon(1/self.rate)
       self.Con = Constant(1) 
       super(Poisson_process,self).__init__(J=self.Con, T=self.Exp, startTime = startTime, startPosition = startPosition)
       self.Poi = stats.poisson
       if ( endTime != None ) and ( endPosition != None ):
            assert endTime > startTime, "invalid times: endTime > startTime."
            self.endTime = endTime
            self.endPosition = endPosition
            self.condition = True
            self.Bin = stats.binom
       elif  ( endTime != None ) != ( endPosition != None ):
            raise Exception( "invalid parameter:", "Must include both endTime AND endPosition or neither" )
  
    def _mean(self,
              t):
        """
        Return the mean at a given time.
        Recall that a conditional Poisson process N_t | N_T=n ~ Bin(n, t/T).
        """
        if not self.conditional:  
            return self.startPosition + self.rate*(t-self.startTime)
        else:
            return self.endPosition*float(t)/self.endTime
    
    def _var(self,
             t):
        """
        Variance at the time point. 
        Recall that a conditional Poisson process N_t | N_T=n ~ Bin(n, t/T)
        """
        if self.conditional:
            return self.endPosition*(1-float(t)/self.endTime)*float(t)/self.endTime
        else:
            return self.rate*(t-self.startTime) 
       
class mpp(p):
    """
    This class constructs marked Poisson process i.e. at exponentially distributed times, a 
    Uniform(L, U) is generated (lower and upper bound of uniform random variate generation).
    There are no other time-space constraints besides startTime.
    """
    def __init__(self,
                 rate = 1, 
                 L = 0, 
                 U = 1, 
                 startTime = 0):
        self.Poi = stats.poisson
        self.L = L
        self.U = U
        self.startTime = startTime
        self.rate = rate
        
    def mpp(self,T):
        """
        Generate the marked Poisson process.
        """      
        p = self.Poi.rvs(self.rate*(T - self.startTime))
        times = self.startTime + (T - self.startTime) * np.random.random(p)
        path = self.L + (self.U - self.L) * np.random.random(p)
        return (path, times) 
