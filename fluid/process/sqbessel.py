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

    def _transition_pdf(self,
                        x,
                        t,
                        y):
        """ 
        Transition from x to y.
        """
        try:
            return (y/x)**(0.5*self.mu)*np.exp(-0.5*(x+y)/self.nu**2/t)/(0.5*self.nu**2*t)*iv(abs(self.mu),4*np.sqrt(x*y)/(self.nu**2*t))
        except AttributeError:
            print("Attn: nu must be known and defined to calculate the transition pdf.")

    def _generate_sample_path_no_absorption(self, 
                                            times):
        """
        Create a smple path without absorption.
        mu must be greater than -1. The parameter times is a list of times to sample at.
        """
        if self.mu<=-1:
            print("Attn: mu must be greater than -1. It is currently %f."%self.mu)
            return
        else:
            if not self.conditional:
                x=self.startPosition
                t=self.startTime
                path=[]
                for time in times:
                    delta=float(time-t)
                    try:
                        y=self.Poi.rvs(0.5*x/delta)
                        x=self.Gamma.rvs(y+self.mu+1)*2*delta
                    except:
                        pass
                    path.append((time,x))
                    t=time
            else:
                path = bridge_creation(self, times, 0)
                return path
            return [(p[0],self.rescalePath(p[1])) for p in path]
         
