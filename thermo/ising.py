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
