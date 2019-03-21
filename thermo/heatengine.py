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
