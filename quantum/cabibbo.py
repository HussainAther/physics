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

# in which dn is d'

# from decay probabilities |V_ud|^2 and |V_us|^2. this method of calculation is more accepted.

V_ud = .22534 # down quark into up quark
V_us = .97427 # strange quark into up quark

c = np.arctan(abs(V_ud)/abs(V_us))

# this gives us two equations
# dn = np.cos(c)*d + np.sin(c)*s and
# sn = -np.sin(c)*d + np.cos(c)*s
# in which sn is s'

"""
We can determine the Cabibbo-Kobayashi-Maskawa matrix to keep track of the weak decays of three
generations of quarks. This allows for CP-violation.
"""
V_ub = .00351 # bottom to up

V_cd = .22520 # down to charm
V_cs = .97344 # spin to charm
V_cb = .0412 # bottom to charm

V_td = .00867 # down to top

ckm_m = np.array([V_ud, V_us, V_ub], [V_cd, V_cs, V_cb], [V_td, V_ts, V_tb])
