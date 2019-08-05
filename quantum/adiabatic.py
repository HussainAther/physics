import numpy as np
import dimod
import matplotlib.pyplot as plt
import minorminer
import networkx as nx
import dwave_networkx as dnx

"""
Adiabatic quantum computing and adiabatic theorem
"""

np.set_printoptions(precision=3, suppress=True)

X = np.array([[0, 1], [1, 0]])
IX = np.kron(np.eye(2), X)
XI = np.kron(X, np.eye(2))
H_0 = - (IX + XI)
λ, v = np.linalg.eigh(H_0)
print("Eigenvalues:", λ)
print("Eigenstate for lowest eigenvalue", v[:, 0])

# Annealing
J = {(0, 1): 1.0, (1, 2): -1.0}
h = {0:0, 1:0, 2:0}
model = dimod.BinaryQuadraticModel(h, J, 0.0, dimod.SPIN)
sampler = dimod.SimulatedAnnealingSampler()
response = sampler.sample(model, num_reads=10)
print("Energy of samples:")
print([solution.energy for solution in response.data()])

# Chimera graph for connectivity structure
connectivity_structure = dnx.chimera_graph(2, 2)
dnx.draw_chimera(connectivity_structure)
plt.show()

G = nx.complete_graph(9)
plt.axis("off")
nx.draw_networkx(G, with_labels=False)
embedded_graph = minorminer.find_embedding(G.edges(), connectivity_structure.edges())
