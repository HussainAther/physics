import numpy as np
import matplotlib.pyplot as plt

from constants import constants
from new_thermo import findTdwv, thetaep, tinvert_thetae, wsat, convertTempToSkew
from convecSkew import convecSkew

"""
Keep track of equilibrium values as a heat engine expands
along the length pvec.
"""

c = constants()

eqT_bot = 30 + c.Tc
eqwv_bot = 14*1.e-3
sfT_bot = 21 + c.Tc
sfwv_bot = 3.e-3
eqwv_top = 3.e-3
sfwv_top = 3.e-3
ptop = 4 10.e2
pbot = 1000.e2

# Perform the functions as the heat expands
eqTd_bot = findTdwv(eqwv_bot,pbot)
sfTd_bot = findTdwv(sfwv_bot,pbot)
thetae_eq = thetaep(eqTd_bot,eqT_bot,pbot)
thetae_sf = thetaep(sfTd_bot,sfT_bot,pbot)

# plot
fig1 = plt.figure(1)
skew, ax1 = convecSkew(1)

# initialize vector arrays
pvec = np.arange(ptop, pbot, 1000)
Tvec_eq = np.zeros(pvec.size)
Tvec_sf = np.zeros(pvec.size)
wv = np.zeros(pvec.size)
wl = np.zeros(pvec.size)
xcoord_eq = np.zeros(pvec.size)
xcoord_sf = np.zeros(pvec.size)
