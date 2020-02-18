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
    
    """
