import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import animation
from numpy.fft import fft2, ifft2, fftshift
from scipy.special import hermite

"""
Bose-Einstein condensate (bose einstein BEC) for evolving Gross-Pitaevskii
equation (gross pitaevskii gpe). 
"""

class QHO:
    """
    Quantum harmonic oscillator (qho) wavefunctions.
    """
    def __init__(self, n, xshift=0, yshift=0):
        self.n = n
        self.xshift = xshift
        self.yshift = yshift
        self.E = n + 0.5
        self.coef = 1 / np.sqrt(2**n * np.factorial(n)) * (1 / np.pi)**(1/4)
        self.hermite = hermite(n)

    def __call__(self, x, y, t):
        xs = x - self.xshift
        ys = y - self.yshift
        return self.coef * np.exp(-(xs**2 + ys**2) / 2 - 1j*self.E*t) * self.hermite(x) * self.hermite(y)

class Simulation:
    """
    Simulation to step wavefunction forward in time from the given parameters
    xmax : maximum extent of boundary
    N    : number of spatial points
    init : initial wavefunction
    nonlinearity : factor in front of |psi^2| term
    """
    def __init__(self, parameters):
        self.parameters = parameters
        
        # Set up spatial dimensions.
        xmax = parameters["xmax"]
        self.xmax = xmax
        N = parameters["N"]
        v = linspace(-xmax, xmax, N)
        self.dx = v[1] - v[0]
        self.x, self.y = np.meshgrid(v, v)

        # Spectral space
        kmax = 2*np.pi / self.dx
        dk = kmax / N
        self.k = np.fft.fftshift((arange(N)-N/2) * dk)
        kx, ky = np.meshgrid(self.k, self.k)

        # Time
        self.steps = 0
        self.time = 0
        self.dt = self.dx**2 / 4

        # Wavefunction
        init_func = parameters["initial"]
        self.wf = init_func(self.x, self.y, 0)
        self.wf /= np.sqrt(self.norm().sum() * self.dx**2) # Normalize

        # Hamiltonian operators
        self.loss = 1 - 1j*parameters["loss"]
        self.T = np.exp(-1j * self.loss * (kx**2 + ky**2) * self.dt / 2)
        self.V = np.exp(-1j * self.loss * (self.x**2 + self.y**2) * self.dt / 2)
        self.eta = parameters["nonlinearity"]

    def evolve(self, time):
        """
        Evolve the wavefunction to the given time in the future.
        """
        steps = int(time / self.dt)
        if steps == 0:
            steps = 1 # Guarantee at least 1 step.

        for _ in range(steps):
            #self.linear_step()
            self.nonlinear_step()
            if self.loss:
                N = self.norm().sum()*self.dx**2
                self.wf /= N

        self.update_time(steps)

     def linear_step(self):
        """
        Make one linear step dt forward in time.
        """
        # Kinetic
        self.wf[:] = np.fft.fft2(np.fft.ifft2(self.wf) * self.T)

        # Potential
        self.wf *= self.V
