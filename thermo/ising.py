import numpy as np

"""
Ising model lattice encapsulation
"""

class IsingLattice:
    """
    Explore a simple nearest-neighbor 2D Ising model.
    This is an NxN periodic lattice in which each site
    has a particle with spin S (+/- 1). You can write the
    energy of the system as 
    E = -J Sum_{i,j} [S_{i,j}*(S_{I+1,j}+S_{i,j+1})]
        -H Sum_{i,j} [S_{i,j}]
    The periodic identification is the  site [i,j] = site [i+m*N,j+l*N] 
    for integers m,l. 
    """
    
    def __init__(self,N,J=1.0,H=0.0):
        self._N = N # lattice dimension
        self._J = J # interaction energy
        self._H = H # external magnetic field
        self.alignedspins() # set all spins to S (+/- 1)
        self._computeEM() # compute E and M over the lattice

    def alignedspins(self, S=1):
        """
        Set all spins to S (+/- 1)
        """
        if not(S==1 or S==-1):
            print("Error: spin must be +/-1")
            raise(AttributeError)
        self._spins = np.ones((self._N, self._N), dtype=int)*S
        self_computeEM()  

    def randomspins(self):
        """
        Randomize the spins.
        """
        self._spins = np.where(np.random.random((self._N, self._N)) > .5, 1, -1)
        self._computeEM()

    def N(self):
        """
        Return lattice dimension.
        """
        return self._N
    
    def M(self):
        """
        Return net magnetization.
        """
        return self._M
     
    def H(self):
        """
        Return exeternal magnetic field.
        """
        return self._H
    
    def J(self):
        """
        Return the interaction energy.
        """
        return self._J
    
    def E(self):
        """
        Return lattice energy.
        """
        return self._E

    def spins(self):
        """
        The following is a "trick" to help maintain the privacy of _spins.
        _spins is mutable, and the "return" without the ()*1 will simply
        return a pointer to _spins, and so could be changed external to the
        class. By multiplying by 1, a new array is created, and that is returned.
        """
        return (self._spins)*1
 
    def spinij(self, i, j):
        """
        Return spin of a specific point (i, j).
        """
        return self._spins[i%self._N, j%self._N]

    def __str__(self):
        """
        Query and return the lattice properties.
        """
        return "\nLattice properties: %d^2 cells, E=%f, M=%d, <E>=%f, <M>=%f\n"%\
               (self._N,self._E,self._M,self._E/self._N**2,self._M/self._N**2)

