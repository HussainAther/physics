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
        
def bridge_creation(process, times, *args):
    """
    Create a bridge between processes.
    """
    # this algorithm relies on the fact that 1-dim diffusion are time reversible.
    print("Attn: using an AR method...")
    process.conditional = False
    temp = process.startPosition
    while True:
        sample_path=[]
        forward = process.generate_sample_path(times, *args)
        process.startPosition = process.endPosition
        backward = process.generate_sample_path(reverse_times(process, times), *args)
        process.startPosition = temp
        check = (forward[0][1]-backward[-1][1]>0)
        i=1
        N = len(times)
        sample_path.append(forward[0])
        while (i<N-1) and (check == (forward[i][1]-backward[-1-i][1]>0) ):
            sample_path.append(forward[i])
            i+=1
        
        if i != N-1: #an intersection was found
            k=0
            while(N-1-i-k>=0):
                sample_path.append((times[i+k],backward[N-1-i-k][1]))
                k+=1
            process.conditional = True
            return sample_path
                 
