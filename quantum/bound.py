from numpy import *
from numpy.linagl import *

"""
Solve the Lippmann-Schwinger integral equation for bound states within a delta-shell
potential. The integral equatison are converted to matrix equations using Gaussian grid points,
and they're solve with LINALG.
"""
