import matplotlib.pyplot as plt
import numpy as np
import dimod

"""
Probabilistic graph models.
"""

n_spins = 3
h = {v: 1 for v in range(n_spins)}
J = {(0, 1): 2,
     (1, 2): -1}
model = dimod.BinaryQuadraticModel(h, J, 0.0, dimod.SPIN)
sampler = dimod.SimulatedAnnealingSampler()
