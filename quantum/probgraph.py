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

# Extract probability
temperature = 1
response = sampler.sample(model, beta_range=[1/temperature, 1/temperature], num_reads=100)

g = {} # dictionary that associate to each energy E the degeneracy g[E]
for solution in response.aggregate().data():
    if solution.energy in g.keys():
        g[solution.energy] += 1
    else:
        g[solution.energy] = 1
print("Degeneracy", g)
probabilities = np.array([g[E] * np.exp(-E/temperature) for E in g.keys()])
Z = probabilities.sum()
probabilities /= Z
fig, ax = plt.subplots()
ax.plot([E for E in g.keys()], probabilities, linewidth=3)
ax.set_xlim(min(g.keys()), max(g.keys()))
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("Energy")
ax.set_ylabel("Probability")
plt.show()
