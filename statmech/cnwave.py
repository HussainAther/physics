import numpy as np

"""
Crank-Nicholson scheme to solve a 2D nonlinear wave equation using methods
from nonlinear dynamics

-d^2phi(x,t)/dt^2 + d^2 phi(x,t)/dx^2 = F(phi) 

Convert the eqution to first-order with
dphi(x,t)/dt - pi(xi,t) = 0
dphi(xi,t)/dt - d^2phi(x,t)/dx^2 + F(phi) = 0
"""

class cnwaveeq:
    """
    Methods for the conversion shown above.
    """
    def __init__(self, F, dF, args, xmin, xmax, Nx):
        """
        F(phi, args) = is a user-specified function returning F(phi) 
        dF(phi, args) = is a user-specified function returning dF/dphi(phi)
        xmin, xmax define the spatial integration domain
        Nx = is the number of points, so dx=(xmax-xmin)/(N-1)
        All initial data set to zero, t set to 0.
        """
        self._F = F
        self._dF = dF
        self._args = args
        self._xmin = xmin
        self._xmax = xmax
        if (Nx <= 1):
            print("Error ... Nx must be greater than 1")
            raise("invalid arguments")
        self._N = Nx
        self._dx = (xmax-xmin)/(Nx-1)
        self._phi_n = np.zeros(Nx)
        self._phi_np1 = np.zeros(Nx)
        self._pi_n = np.zeros(Nx)
        self._pi_np1 = np.zeros(Nx)
        self._phi_res = np.zeros(Nx)
        self._pi_res = np.zeros(Nx)
        self._F_n = np.zeros(Nx)
        self._F_np1 = np.zeros(Nx)
        self._dF_n = np.zeros(Nx)
        self._dF_np1 = np.zeros(Nx)
        self._t = 0
        self._tol = 1.0e-4
        self._max_iter = 50
        self._dt = 0.5*self._dx
        self._eps = 0 

    def set_phi_n(self, phi_init, args):
        """
        Set the initial data phi(x,t=0) to a lsit of parameters args with the initial
        condition phi_init.
        """
        self._phi_n = phi_init(self.x(),args)
        # Enforce periodicity.
        self._phi_n[self._N-1] = self._phi_n[0]

    def set_pi_n(self, pi_init, args):
        """
        Same but for pi.
        """
        self._pi_n = pi_init(self.x(),args)
        # Enforce periodicity.
        self._pi_n[self._N-1] = self._pi_n[0]

    def set_CFL(self,CFL):
        """
        Set the convergence condition by Courant–Friedrichs–Lewy. 
        """
        self._dt = CFL*self._dx

    def set_iter(self,tol,max_iter):
        """
        Set the parameters the interative solution method will iterate
        until a norm of the residual is below tol, up to a maximum of max_iter 
        iterations
        """
        self._tol = tol
        self._max_iter = max_iter

    def set_KO_filter(self,eps):
        """
        Set Kreiss-Oliger dissipation parameter eps (<1). 
        """
        if (eps < 0 or eps > 1):
            raise("set_KO_filter: error: eps out of range")
        self._eps = eps

    def step(self):
        """
        Take 1 time step of size dt. The equations are solved iteratively,
        and the iterativion proceeds until the residual is below tol,
        unless max_iter is exceeded. After the time step, time levels
        np1 and n are swapped, so level n becomes the current (latest)
        time, and np1 holds the past time level data.
        
        Return  [iter,tol], the number of iterations required, and pre-last step
        residual
        """
        N = self._N
        dx = self._dx
        dt = self._dt
        CFL = dt/dx
        # filter if desired
        if (self._eps >= 0):
            self._phi_n = KO_filter(self._phi_n,self._eps)
            self._pi_n = KO_filter(self._pi_n,self._eps)
        # initial guess ... copy N to N+1
        self._phi_np1 = self._phi_n[:]
        self._pi_np1 = self._pi_n[:]
