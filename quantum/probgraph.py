import matplotlib.pyplot as plt
import numpy as np
import dimod

from dimod.reference.samplers.simulated_annealing import greedy_coloring

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

clamped_spins = {0: -1}
num_sweeps = 1000
betas = [1.0 - 1.0*i / (num_sweeps - 1.) for i in range(num_sweeps)]

# Set up the adjacency matrix.
adj = {n: set() for n in h}
for n0, n1 in J:
    adj[n0].add(n1)
    adj[n1].add(n0)

# Use a vertex coloring for the graph and update the nodes by color
__, colors = greedy_coloring(adj)

spins = {v: np.random.choice((-1, 1)) if v not in clamped_spins else clamped_spins[v]
         for v in h}

for beta in betas:
    energy_diff_h = {v: -2 * spins[v] * h[v] for v in h}
    # for each color, do updates
    for color in colors:
        nodes = colors[color]
        energy_diff_J = {}
        for v0 in nodes:
            ediff = 0
            for v1 in adj[v0]:
                if (v0, v1) in J:
                    ediff += spins[v0] * spins[v1] * J[(v0, v1)]
                if (v1, v0) in J:
                    ediff += spins[v0] * spins[v1] * J[(v1, v0)]
            energy_diff_J[v0] = -2. * ediff
        for v in filter(lambda x: x not in clamped_spins, nodes):
            logp = np.log(np.random.uniform(0, 1))
            if logp < -1. * beta * (energy_diff_h[v] + energy_diff_J[v]):
                spins[v] *= -1
spins
