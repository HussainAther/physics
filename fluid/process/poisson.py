import numpy as np

from scipy.stats import stats

class poissonprocess()
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
   
