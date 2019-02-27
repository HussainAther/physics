import random
from vpython.graph import *
import numpy as np

"""
Determine the ground-state probabilty via a Feynman path integration
using the Metropolis algorithm to simulate variations about the classical trajectory
"""

N = 100
M = 101
xscale = 10
path = np.zeros([M], float)
prob = np.zeros([M], float)

trajec = display(width=300, height=500, title="Spacetime Trajectories")
trplot = curve(y=range(0, 100), color=color.magenta, display=trajec)

def trjaxs():
    trax = curve(pos=[(-97, -100), (100, -100)], color=color.cyan, display = trajec)
    label(pos = (0,−110), text = ’0’, box = 0, display = trajec)
    label(pos = (60,−110), text = ’x’, box = 0, display = trajec)
