import numpy as np
import matplotlib.pyplot as plt
from constants import constants
from new_thermo import findTdwv, thetaep, tinvert_thetae, wsat, convertTempToSkew
from convecSkew import convecSkew

"""
Keep track of equilibrium values as a heat engine expands
along the length pvec.
"""
