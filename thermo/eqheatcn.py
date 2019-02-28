import matplotlib.pylab as p
from mpl_toolkits.mplot3d import Axe3D
from vpython import *
import numpy as np

"""
Solve the heat equation in one dimension adn time using the Crank-Nicolson method.
Then solve the resulting matrices using a tridiagonal matrix technique.
"""

Max = 51
n = 50
m = 50
Ta = np.zeros((Max), float)
Tb = np.zeros((Max), float)
Tc = np.zeros((Max), float)
a = np.zeros((Max), float)
b = np.zeros((Max), float)
c = np.zeros((Max), float)
d = np.zeros((Max), float)
x = np.zeros((Max), float)
t = np.zeros((Max, Max), float)
