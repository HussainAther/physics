import numpy as np
import matplotlib.pyplot as plt

"""
Physicist Nicola Cabibbo introduced the Cabibbo angle to preserve the
universality of the weak interaction.
"""

d = np.array([3, 4]) # some vector
s = np.array([0, 4]) # some other vector
c = np.angle(45) # rotate it 45 deg. this is the Cabibbo angle. It represents the rotation of the mass
                     # eigenstate vector space formed by the mas  eigenstates |d> and |s> into the weak eigenstate
                     # vector space formed by the weak eigenstate |d'> and |s'>
dn = np.cos(c)*d + np.sin(c)*s

# from decay probabilities |V_d|^2 and |V_s|^2. this method of calculation is more accepted.

V_d = .22534 # down quark into up quark
V_s = .97427 # strange quark into up quark

c = tanh

