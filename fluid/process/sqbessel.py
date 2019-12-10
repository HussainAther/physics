import numpy as np
import scipy.stats as stats

class SqBessel_process(p):
    """
    The (lambda0 dimensional) squared Bessel process is defined by the SDE:
    dX_t = lambda_0*dt + nu*sqrt(X_t)dB_t
    Based on R.N. Makarov and D. Glew's research on simulating squared bessel process. See "Exact
    Simulation of Bessel Diffusions", 2011.
    """
    def __init__(self, 
                 lambda_0, 
                 nu, 
                 startTime = 0, 
                 startPosition = 1, 
                 endTime = None, 
                 endPosition = None):
            super(SqBessel_process, self).__init__(startTime, startPosition, endTime, endPosition)
            try:
                self.endPosition = 4.0/parameters["nu"]**2*self.endPosition
                self.x_T = self.endPosition
            except:
                pass
            self.x_0 = 4.0/nu**2*self.startPosition
            self.nu = nu
            self.lambda0 = lambda_0
            self.mu = 2*float(self.lambda0)/(self.nu*self.nu)-1
            self.Poi = stats.poisson
            self.Gamma = stats.gamma
            self.Nor = stats.norm
            self.InGamma = IncompleteGamma

    def generate_sample_path(self,
                             times,
                             absb=0):
        """
        absb is a boolean for the square bessel function, true if absorbtion at 0, false else. 
        """
        if absb:
            return self._generate_sample_path_with_absorption(times)
        else:
            return self._generate_sample_path_no_absorption(times)
